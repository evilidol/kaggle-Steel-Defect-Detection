#https://github.com/junfu1115/DANet

from common  import *
from dataset import *
from resnet  import *


####################################################################################################
def upsize(x,scale_factor=2):
    #x = F.interpolate(x, size=e.shape[2:], mode='nearest')
    x = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
    return x

class Decode(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Decode, self).__init__()

        self.top = nn.Sequential(
            nn.Conv2d(in_channel, out_channel//2, kernel_size=3,  stride=1, padding=1, bias=False),
            BatchNorm2d( out_channel//2),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.1),

            nn.Conv2d(out_channel//2, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.1),

            # nn.Conv2d(out_channel//2, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            # BatchNorm2d(out_channel),
            # nn.ReLU(inplace=True), #Swish(), #
        )

    def forward(self, x):
        x = self.top(torch.cat(x, 1))
        return x

class Net(nn.Module):

    def load_pretrain(self, skip, is_print=True):
        load_pretrain(self, skip, pretrain_file=PRETRAIN_FILE, conversion=CONVERSION, is_print=is_print)

    def __init__(self, num_class=4, drop_connect_rate=0.2):
        super(Net, self).__init__()

        e = ResNet34()
        self.block0 = e.block0  # 64, 128, 128
        self.block1 = e.block1  # 64,  64,  64
        self.block2 = e.block2  #128,  32,  32
        self.block3 = e.block3  #256,  16,  16
        self.block4 = e.block4  #512,   8,   8
        e = None  #dropped

        self.decode1 = Decode(512,     256) # 8,   8
        self.decode2 = Decode(256+256, 256) #16,  16
        self.decode3 = Decode(256+128, 128)
        self.decode4 = Decode(128+ 64,  64)
        self.decode5 = Decode( 64+ 64,  64)
        self.decode6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.logit = nn.Conv2d(64,num_class+1, kernel_size=1)

    def forward(self, x):
        batch_size,C,H,W = x.shape

        #----------------------------------
        backbone = []
        x = self.block0(x) ; backbone.append(x)
        x = self.block1(x) ; backbone.append(x)
        x = self.block2(x) ; backbone.append(x)
        x = self.block3(x) ; backbone.append(x)
        x = self.block4(x) ; backbone.append(x)

        #----------------------------------
        x = self.decode1([backbone[-1], ])                   #; print('d1',d1.size())
        x = self.decode2([backbone[-2], upsize(x)])          #; print('d2',d2.size())
        x = self.decode3([backbone[-3], upsize(x)])          #; print('d3',d3.size())
        x = self.decode4([backbone[-4], upsize(x)])          #; print('d4',d4.size())
        x = self.decode5([backbone[-5], upsize(x)])          #; print('d5',d5.size())
        x = self.decode6(upsize(x))
        logit = self.logit(x)
        return logit


### loss ###################################################################

def criterion(logit, truth, weight=None):
    logit = logit.permute(0, 2, 3, 1).contiguous().view(-1, 5)
    truth = truth.permute(0, 2, 3, 1).contiguous().view(-1)

    if weight is not None:
        weight = torch.FloatTensor([1]+ weight).to(truth.device) #4 pos + one background class
    loss = F.cross_entropy(logit, truth, weight=weight, reduction='none')

    loss = loss.mean()
    return loss



#----

def metric_hit(logit, truth, threshold=0.5):
    batch_size,num_class, H,W = logit.shape

    with torch.no_grad():
        logit = logit.view(batch_size,num_class,-1)
        truth = truth.view(batch_size,-1)

        probability = torch.softmax(logit,1)
        p = torch.max(probability, 1)[1]
        t = truth
        correct = (p==t)

        index0 = t==0
        index1 = t==1
        index2 = t==2
        index3 = t==3
        index4 = t==4

        num_neg  = index0.sum().item()
        num_pos1 = index1.sum().item()
        num_pos2 = index2.sum().item()
        num_pos3 = index3.sum().item()
        num_pos4 = index4.sum().item()

        neg  = correct[index0].sum().item()/(num_neg +1e-12)
        pos1 = correct[index1].sum().item()/(num_pos1+1e-12)
        pos2 = correct[index2].sum().item()/(num_pos2+1e-12)
        pos3 = correct[index3].sum().item()/(num_pos3+1e-12)
        pos4 = correct[index4].sum().item()/(num_pos4+1e-12)

        num_pos = [num_pos1,num_pos2,num_pos3,num_pos4,]
        tn = neg
        tp = [pos1,pos2,pos3,pos4,]

    return tn,tp, num_neg,num_pos



def metric_dice(logit, truth, threshold=0.1, sum_threshold=1):

    with torch.no_grad():
        probability = torch.softmax(logit,1)
        probability = predict_to_onehot(probability)
        truth = index_to_onehot(truth)

        batch_size,num_class, H,W = truth.shape
        probability = probability.view(batch_size,num_class,-1)
        truth = truth.view(batch_size,num_class,-1)
        p = (probability>threshold).float()
        t = (truth>0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)

        d_neg = (p_sum < sum_threshold).float()
        d_pos = 2*(p*t).sum(-1)/((p+t).sum(-1)+1e-12)

        neg_index = (t_sum==0).float()
        pos_index = 1-neg_index

        num_neg = neg_index.sum()
        num_pos = pos_index.sum(0)
        dn = (neg_index*d_neg).sum()/(num_neg+1e-12)
        dp = (pos_index*d_pos).sum(0)/(num_pos+1e-12)

        #----

        dn = dn.item()
        dp = list(dp.data.cpu().numpy())
        num_neg = num_neg.item()
        num_pos = list(num_pos.data.cpu().numpy())

    return dn,dp, num_neg,num_pos


##############################################################################################
def make_dummy_data(folder='256x256', batch_size=8):

    image_file =  glob.glob('/root/share/project/kaggle/2019/steel/data/dump/%s/image/*.png'%folder) #32
    image_file = sorted(image_file)

    input=[]
    truth_mask =[]
    truth_label=[]
    infor= None

    for b in range(0, batch_size):
        i = b%len(image_file)
        image = cv2.imread(image_file[i], cv2.IMREAD_COLOR)
        mask  = np.load(image_file[i].replace('/image/','/mask/').replace('.png','.npy'))

        input.append(image)
        truth_mask.append(mask)

    input = np.array(input)
    input = image_to_input(input, IMAGE_RGB_MEAN, IMAGE_RGB_STD)
    input = torch.from_numpy(input).float().cuda()

    truth_mask = np.array(truth_mask)
    truth_mask = (truth_mask>0).astype(np.float32)
    truth_mask = torch.from_numpy(truth_mask).float().cuda()

    #---
    #conversion
    with torch.no_grad():
        truth_label = mask_to_label(truth_mask).float()
        truth_mask  = onehot_to_index(truth_mask)

    return input, truth_mask, truth_label, infor


#########################################################################
def run_check_basenet():
    net = Net()
    #print(net)
    net.load_pretrain(skip=['logit'])

   #---
    if 0:
        print(net)
        print('')

        print('*** print key *** ')
        state_dict = net.state_dict()
        keys = list(state_dict.keys())
        #keys = sorted(keys)
        for k in keys:
            if any(s in k for s in [
                'num_batches_tracked'
                # '.kernel',
                # '.gamma',
                # '.beta',
                # '.running_mean',
                # '.running_var',
            ]):
                continue

            p = state_dict[k].data.cpu().numpy()
            print(' \'%s\',\t%s,'%(k,tuple(p.shape)))
        print('')
        exit(0)

def run_check_net():

    batch_size = 1
    C, H, W    = 3, 256, 1600

    input = np.random.uniform(-1,1,(batch_size,C, H, W ))
    input = np.random.uniform(-1,1,(batch_size,C, H, W ))
    input = torch.from_numpy(input).float().cuda()

    net = Net().cuda()
    net.eval()

    with torch.no_grad():
        logit = net(input)

    print('')
    print('input: ',input.shape)
    print('logit: ',logit.shape)
    #print(net)


def run_check_train():

    loss_weight = [ 5,1,1,1 ]

    if 1:
        input, truth_mask, truth_label, infor = make_dummy_data(folder='256x256', batch_size=20)
        batch_size, C, H, W  = input.shape

        print('input: ',input.shape)
        print('truth_label: ',truth_label.shape)
        print('(count)    : ',truth_label.sum(0))
        print('truth_mask: ',truth_mask.shape)
        print('')

    #---


    net = Net(drop_connect_rate=0.1).cuda()
    net.load_pretrain(skip=['logit'],is_print=False)#

    net = net.eval()
    with torch.no_grad():
        logit = net(input)
        print('input: ',input.shape)
        print('logit: ',logit.shape)
        print('')

        loss = criterion(logit, truth_mask, loss_weight)
        tn,tp, num_neg,num_pos = metric_hit (logit, truth_mask)

        print('loss = %0.5f'%loss.item())
        print('tn,tp = %0.5f, [%0.5f,%0.5f,%0.5f,%0.5f] '%(tn,tp[0],tp[1],tp[2],tp[3]))
        print('num_pos,num_neg = %d, [%d,%d,%d,%d] '%(num_neg,num_pos[0],num_pos[1],num_pos[2],num_pos[3]))
        print('')


    #exit(0)
    # dummy sgd to see if it can converge ...
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                      lr=0.1, momentum=0.9, weight_decay=0.0001)

    #optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=0.001)

    print('batch_size =',batch_size)
    print('--------------------------------------------')
    print('[iter ]  loss     |  tn, [tp1,tp2,tp3,tp4]  ')
    print('--------------------------------------------')
          #[00000]  0.70383  | 0.00000, 0.46449


    i=0
    optimizer.zero_grad()
    while i<=200:

        net.train()
        optimizer.zero_grad()

        logit = net(input)
        loss  = criterion(logit, truth_mask, loss_weight)

        tn,tp, num_neg,num_pos = metric_hit(logit, truth_mask)

        (loss).backward()
        optimizer.step()

        if i%10==0:
            print('[%05d] %8.5f  | %0.5f, [%0.5f,%0.5f,%0.5f,%0.5f]  '%(
                i,
                loss.item(),
                tn,tp[0],tp[1],tp[2],tp[3],
            ))
        i = i+1
    print('')


    if 1:
        #net.eval()
        logit = net(input)
        probability = torch.softmax(logit,1)
        probability = predict_to_onehot(probability)
        truth_mask  = index_to_onehot(truth_mask)

        probability_mask = probability.data.cpu().numpy()
        truth_label = truth_label.data.cpu().numpy()
        truth_mask  = truth_mask.data.cpu().numpy()
        image = input_to_image(input, IMAGE_RGB_MEAN,IMAGE_RGB_STD)


        for b in range(batch_size):
            print('%2d ------ '%(b))
            result = draw_predict_result(image[b], truth_mask[b], truth_label[b], probability_mask[b])
            image_show('result',result, resize=0.5)
            cv2.waitKey(0)



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #run_check_basenet()
    #run_check_net()
    run_check_train()


    print('\nsucess!')


