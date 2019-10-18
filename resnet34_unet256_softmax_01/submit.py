import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from common  import *
from dataset import *
from model   import *
from train_b0   import *


######################################################################################
from metric import *


TEMPERATE=0.5
######################################################################################



def compute_metric(truth, predict):

    num = len(truth)
    t = truth.reshape(num*4,-1)
    p = predict.reshape(num*4,-1)
    t_sum = t.sum(-1)
    p_sum = p.sum(-1)
    pt_sum = (p*t).sum(-1)

    h_neg = (p_sum == 0).astype(np.float32)
    h_pos = (p_sum >  0).astype(np.float32)
    d_pos = 2* pt_sum/(t_sum+p_sum+1e-12)


    t_sum = t_sum.reshape(num,4)
    p_sum = p_sum.reshape(num,4)
    h_neg = h_neg.reshape(num,4)
    h_pos = h_pos.reshape(num,4)
    d_pos = d_pos.reshape(num,4)

    #for each class
    hit_neg = []
    hit_pos = []
    dice_pos = []
    for c in range(4):
        neg_index = np.where(t_sum[:,c]==0)[0]
        pos_index = np.where(t_sum[:,c]>=1)[0]
        hit_neg.append(h_neg[:,c][neg_index])
        hit_pos.append(h_pos[:,c][pos_index])
        dice_pos.append(d_pos[:,c][pos_index])


    ##
    hit_neg_all = np.concatenate(hit_neg).mean()
    hit_pos_all = np.concatenate(hit_pos).mean()
    hit_neg  = [np.nan_to_num(h.mean(),0) for h in hit_neg]
    hit_pos  = [np.nan_to_num(h.mean(),0) for h in hit_pos]
    dice_pos = [np.nan_to_num(d.mean(),0) for d in dice_pos]


    ## from kaggle probing ...
    kaggle_pos = np.array([ 128,43,741,120 ])
    kaggle_neg_all = 6172
    kaggle_all     = 1801*4
    kaggle = (hit_neg_all*kaggle_neg_all + sum(dice_pos*kaggle_pos))/kaggle_all



    confusion = np.zeros((5,5), np.float32)
    if 0:
        #confusion matrix
        t = truth.transpose(1,0,2,3).reshape(4,-1)
        t = np.vstack([t.sum(0,keepdims=True)==0,t])
        p = predict.transpose(0,2,3,1).reshape(-1,4)
        p = np.hstack([p.sum(1,keepdims=True)==0,p])

        confusion = np.zeros((5,5), np.float32)
        for c in range(5):
            index = np.where(t[c]==1)[0]
            confusion[c] = p[index].sum(0)/len(index)


    #print (np.array_str(confusion, precision=3, suppress_small=True))
    return kaggle,hit_neg_all,hit_pos_all,hit_neg,hit_pos,dice_pos,confusion



def do_evaluate_segmentation(net, test_dataset, augment=[], out_dir=None):

    test_loader = DataLoader(
            test_dataset,
            sampler     = SequentialSampler(test_dataset),
            batch_size  = 2,
            drop_last   = False,
            num_workers = 2,
            pin_memory  = True,
            collate_fn  = null_collate
    )
    #----

    #def sharpen(p,t=0):
    def sharpen(p,t=TEMPERATE):
        if t!=0:
            return p**t
        else:
            return p


    test_num  = 0
    test_id   = []
    #test_image = []
    test_probability_label = [] # 8bit
    test_probability_mask  = [] # 8bit
    test_truth_label = []
    test_truth_mask  = []

    start = timer()
    for t, (input, truth_mask, truth_label, infor) in enumerate(test_loader):

        batch_size,C,H,W = input.shape
        input = input.cuda()

        with torch.no_grad():
            net.eval()

            num_augment = 0
            if 1: #  null
                logit =  data_parallel(net,input)  #net(input)
                probability = torch.softmax(logit,1)

                probability_mask = sharpen(probability,0)
                num_augment+=1

            if 'flip_lr' in augment:
                logit = data_parallel(net,torch.flip(input,dims=[3]))
                probability  = torch.softmax(torch.flip(logit,dims=[3]),1)

                probability_mask += sharpen(probability)
                num_augment+=1

            if 'flip_ud' in augment:
                logit = data_parallel(net,torch.flip(input,dims=[2]))
                probability = torch.softmax(torch.flip(logit,dims=[2]),1)

                probability_mask += sharpen(probability)
                num_augment+=1

            #---
            probability_mask = probability_mask/num_augment

        #---
        batch_size  = len(infor)
        truth_label = truth_label.data.cpu().numpy().astype(np.uint8)
        truth_mask  = truth_mask.data.cpu().numpy().astype(np.uint8)
        probability_mask = (probability_mask.data.cpu().numpy()*255).astype(np.uint8)

        test_id.extend([i.image_id for i in infor])
        test_probability_mask.append(probability_mask)
        test_truth_mask.append(truth_mask)
        test_truth_label.append(truth_label)
        test_num += batch_size

        #---
        print('\r %4d / %4d  %s'%(
             test_num, len(test_loader.dataset), time_to_str((timer() - start),'min')
        ),end='',flush=True)

    assert(test_num == len(test_loader.dataset))
    print('')

    start_timer = timer()
    test_probability_mask = np.concatenate(test_probability_mask)
    test_truth_mask  = np.concatenate(test_truth_mask)
    test_truth_label = np.concatenate(test_truth_label)
    print(time_to_str((timer() - start_timer),'sec'))

    return test_id, test_probability_label, test_probability_mask, test_truth_label, test_truth_mask


######################################################################################
def run_submit_segmentation(


):
    out_dir = \
        '/root/share/project/kaggle/2019/steel/result5/resnet34-unet256-fold-0'
    initial_checkpoint = \
        '/root/share/project/kaggle/2019/steel/result5/resnet34-unet256-fold-0/checkpoint/00096000_model.pth'
    train_split   = ['valid0_500.npy',]


    out_dir = \
        '/root/share/project/kaggle/2019/steel/result5/resnet34-unet256-foldb0-0'
    initial_checkpoint = \
        '/root/share/project/kaggle/2019/steel/result5/resnet34-unet256-foldb0-0/checkpoint/00075000_model.pth'
    train_split   = ['valid_b0_1000.npy',]


    # out_dir = \
    #     '/root/share/project/kaggle/2019/steel/result5/resnet34-unet256-foldb1-0'
    # initial_checkpoint = \
    #     '/root/share/project/kaggle/2019/steel/result5/resnet34-unet256-foldb1-0/checkpoint/00123000_model.pth'
    # train_split   = ['valid_b1_1000.npy',]


    mode = 'test' #'train' # 'test'
    augment =  ['null', 'flip_lr','flip_ud'] # ['null, 'flip_lr','flip_ud','5crop']

    #---

    ## setup
    os.makedirs(out_dir +'/submit/%s'%(mode), exist_ok=True)

    log = Logger()
    log.open(out_dir+'/log.submit.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    ## dataset -------

    log.write('** dataset setting **\n')
    if mode == 'train':
        test_dataset = SteelDataset(
            mode    = 'train',
            csv     = ['train.csv',],
            #split   = ['valid0_500.npy',],
            split   = train_split, #['valid_b0_1000.npy',],
            augment = None,
        )

    if mode == 'test':
        test_dataset = SteelDataset(
            mode    = 'test',
            csv     = ['sample_submission.csv',],
            split   = ['test_1801.npy',],
            augment = None, #
        )

    log.write('test_dataset : \n%s\n'%(test_dataset))
    log.write('\n')
    #exit(0)

    ## net ----------------------------------------
    log.write('** net setting **\n')

    net = Net().cuda()
    net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage), strict=False)

    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    log.write('%s\n'%(type(net)))
    log.write('\n')

    ## start testing here! ##############################################
    #

    if 1: #save
        image_id, probability_label, probability_mask, truth_label, truth_mask =\
            do_evaluate_segmentation(net, test_dataset, augment)

    if 1: #save
        write_list_to_file (out_dir + '/submit/%s/image_id.txt'%(mode),image_id)
        np.savez_compressed(out_dir + '/submit/%s/truth_mask.uint8.npz'%(mode), truth_mask)
        np.savez_compressed(out_dir + '/submit/%s/probability_mask.uint8.npz'%(mode), probability_mask)

        #exit(0)

    if 1:
        image_id = read_list_from_file(out_dir + '/submit/%s/image_id.txt'%(mode))
        truth_mask       = np.load(out_dir + '/submit/%s/truth_mask.uint8.npz'%(mode))['arr_0']
        probability_mask = np.load(out_dir + '/submit/%s/probability_mask.uint8.npz'%(mode))['arr_0']


    if 0:
        #do some analysis
        num_test= len(image_id)

        t = truth_mask.reshape(-1)
        p = probability_mask.transpose(0,2,3,1).reshape(-1,5)
        print('xx1')

        auc=[]
        result=[]
        for c in [0,1,2,3,4]:
            print('xx2 @', c)

            tc, pc = t==c, p[:,c]


            #sub sample for speed
            tc, pc = tc[::100], pc[::100]



            a = sklearn_metrics.roc_auc_score(tc, pc)
            auc.append(a)

            fpr, tpr, threshold = sklearn_metrics.roc_curve(tc, pc)
            eer, threshold_eer  = compute_eer(fpr,tpr,threshold)
            result.append([fpr, tpr, threshold])

        text = print_roc_label(auc, result)
        print(text)




        exit(0)

    #----
    if 1: #decode
        num_test= len(image_id)

        value = np.max(probability_mask,1,keepdims=True)
        value = probability_mask*(value==probability_mask)
        probability_mask = probability_mask[:,1:] #remove background class

        index = np.ones((num_test,4,256,1600),np.uint8)*np.array([1,2,3,4],np.uint8).reshape(1,4,1,1)
        truth_mask = truth_mask==index



    if 0:
        num_test= len(image_id)
        #predict_mask = probability_mask>(np.array([0.5,0.5,0.5,0.5])*255).astype(np.uint8).reshape(1,4,1,1)

        for c in [0,1,2,3]:
            best_neg = [0,None]
            best_pos = [0,None]

            for pixel_threshold in [
                [0.25,0.25,0.25,0.25],
                [0.40,0.40,0.40,0.40],
                [0.50,0.50,0.50,0.50],
                [0.60,0.60,0.60,0.60],
                [0.65,0.65,0.65,0.65],
                [0.70,0.70,0.70,0.70],
                [0.75,0.75,0.75,0.75],
                [0.80,0.80,0.80,0.80],
                [0.85,0.85,0.85,0.85],
                [0.90,0.90,0.90,0.90],
                [0.95,0.95,0.95,0.95],
            ]:
                print(' *** grid search @ pixel_threshold=%s ***'%str(pixel_threshold))
                print(' ')

                predict_mask = probability_mask>(np.array(pixel_threshold)*255).astype(np.uint8).reshape(1,4,1,1)

                #---
                base_neg_dice = 0
                collect=[]
                for t,p in zip(truth_mask[:,c],predict_mask[:,c]):
                    t_sum = t.sum()

                    num_component, component = cv2.connectedComponents(p.astype(np.uint8))
                    if num_component==1:
                        if t_sum==0: base_neg_dice+=1
                        continue

                    p_sum = p.sum()
                    i_sum = (t*p).sum()

                    size = np.zeros(num_component-1)
                    intersect = np.zeros(num_component-1)
                    for cnt in range(1,num_component):
                        select = component==cnt
                        size[cnt-1] = p[select].sum()
                        intersect[cnt-1] = t[select].sum()

                    collect.append([t_sum, p_sum, i_sum, intersect, size])

                ##-------
                r = []

                print('class%d @ pixel_threshold=%0.3f'%((c+1),pixel_threshold[c]))
                print('size neg   pos        dice')
                print('--------------------------------')
                      #0	913	  44.257	 957.257
                for threshold in [
                    range(0, 5000, 200),
                    range(0, 5000, 200),
                    range(0, 8000, 500),
                    range(0, 8000, 500),
                ][c]:
                    neg_dice = base_neg_dice
                    pos_dice = 0
                    for t_sum, p_sum, i_sum, intersect, size in collect:

                        for i,s in zip(intersect, size):
                            i_sum -= (s<threshold)*i
                            p_sum -= (s<threshold)*s

                        if t_sum==0:
                            if p_sum==0:
                               neg_dice+=1
                        else:
                               pos_dice+=2*i_sum/(t_sum+p_sum)

                    print('%5d\t%3d\t%8.3f\t%8.3f'%(threshold, neg_dice, pos_dice, neg_dice+pos_dice))
                    r.append([threshold, neg_dice, pos_dice, neg_dice+pos_dice])

                #store best
                # r = np.array(r)
                # argmax = r[:,1].argmax()
                # if best_neg[0]<r[:,1].max():
                #      best_neg[1]
                #      best_neg[0]=(argmax)



                print('')

                zz=0

    if 0:
        num_test= len(image_id)
        predict_mask = probability_mask>(np.array([0.5,0.5,0.5,0.5])*255).astype(np.uint8).reshape(1,4,1,1)

        for c in [0,1,2,3]:

            #size, gain, loss
            collect=[]
            for t,p in zip(truth_mask[:,c],predict_mask[:,c]):
                num_component, component = cv2.connectedComponents(p.astype(np.uint8))
                if num_component==1: continue

                t_sum = t.sum()

                v = []
                for cnt in range(1,num_component):
                    select = component==cnt
                    t_area = t[select].sum()
                    p_area = p[select].sum()

                    v.append([p_area, t_area]) #size, tp, fp

                v = np.array(v)
                v = v[np.argsort(-v[:,0])]  #sort by max area

                #----
                size = v[:,0]
                loss = v[:,0]/(t_sum+1e-8)
                gain = np.zeros(num_component-1)
                if t_sum==0: gain[0]=1

                collect.append(np.stack([size,loss,gain]).T)

            #----
            collect = np.vstack(collect)
            collect = collect[np.argsort(collect[:,0])]

            s    = collect[:,0]
            loss = collect[:,1].cumsum()
            gain = collect[:,2].cumsum()

            plt.plot(s,loss, 'r-')
            plt.plot(s,gain, 'b-')
            #plt.plot(loss,gain, 'k-')
            plt.show()

            zz=0

    if 0:
        num_test= len(image_id)
        predict_mask = probability_mask>(np.array([0.75,0.5,0.5,0.5])*255).astype(np.uint8).reshape(1,4,1,1)

        for c in [0,1,2,3]:
            collect = []

            for t,p in zip(truth_mask[:,c],predict_mask[:,c]):
                num_component, component = cv2.connectedComponents(p.astype(np.uint8))
                t_sum = t.sum()

                if num_component==1: continue
                for cnt in range(1,num_component):
                    select = component==cnt
                    t_area = t[select].sum()
                    p_area = p[select].sum()

                    collect.append([p_area, t_area, p_area-t_area, t_sum]) #size, tp, fp

            collect = np.array(collect)
            argsort = np.argsort(collect[:,0])
            collect = collect[argsort]

            s  = collect[:,0]
            tp = collect[:,1].cumsum()
            fp = collect[:,2].cumsum()
            empty = (collect[:,3]==0).cumsum()

            plt.plot(s,tp, 'r-')
            plt.plot(s,fp, 'b-')
            plt.plot(s,empty, 'k-')
            plt.show()

            zz=0

            #
            #
            #
            #
            #

    # for b in range(len(predict)):
    #     for c in range(4):
    #         predict[b,c] = remove_small_one(predict[b,c], min_size[c])
    # return predict


    #--
    # if 0:
    #     # save for ensembling
    #     probability_mask = (probability_mask*255).astype(np.uint8)
    #     probability_mask[probability_mask<32]=0
    #
    #     probability = (probability*255).astype(np.uint8)
    #     probability[probability<32]=0
    #
    #     truth_mask = (truth_mask*255).astype(np.uint8)
    #
    #     write_list_to_file(out_dir + '/submit/%s/image_id.txt'%(mode),image_id)
    #     np.savez_compressed(out_dir + '/submit/%s/truth_mask.uint8.npz'%(mode), truth_mask)
    #     np.savez_compressed(out_dir + '/submit/%s/probability_mask.uint8.npz'%(mode), probability_mask)
    #     np.savez_compressed(out_dir + '/submit/%s/probability.uint8.npz'%(mode), probability)
    #     #exit(0)



    #---
    threshold_pixel = [0.650,0.650,0.600,0.400,]
    threshold_size  = [400,600,1000,2500,]


    # inspect here !!!  ###################
    print('')
    log.write('submitting .... @ %s\n'%str(augment))
    log.write('threshold_pixel = %s\n'%str(threshold_pixel))
    log.write('threshold_size  = %s\n'%str(threshold_size))
    log.write('\n')

    if mode == 'train':

        #-----
        def log_train_metric():
            log.write('\n')
            log.write('kaggle      = %f\n'%kaggle)
            log.write('hit_neg_all = %f\n'%hit_neg_all)
            log.write('hit_pos_all = %f\n'%hit_pos_all)
            log.write('\n')

            log.write('* image level metric *\n')
            for c in range(4):
                log.write('dice_pos[%d], hit_pos[%d], hit_neg[%d] = %0.5f,  %0.5f,  %0.5f\n'%(
                    c+1,c+1,c+1,dice_pos[c],hit_pos[c],hit_neg[c]
                ))
            log.write('\n')

            log.write('confusion\n')
            log.write('%s\n'%(np.array_str(confusion, precision=3, suppress_small=True)))
            log.write('\n')
        #-----

        log.write('** after threshold_pixel **\n')

        start_timer = timer()
        predict_mask = probability_mask>(np.array(threshold_pixel)*255).astype(np.uint8).reshape(1,4,1,1)
        print(time_to_str((timer() - start_timer),'sec'))

        kaggle,hit_neg_all,hit_pos_all,hit_neg,hit_pos,dice_pos,confusion =\
            compute_metric(truth_mask, predict_mask)

        log_train_metric()

        #-----

        log.write('** after threshold_size **\n')

        start_timer = timer()
        predict_mask = remove_small(predict_mask, threshold_size)
        print(time_to_str((timer() - start_timer),'sec'))

        kaggle,hit_neg_all,hit_pos_all,hit_neg,hit_pos,dice_pos,confusion =\
            compute_metric(truth_mask, predict_mask)

        log_train_metric()


    ###################

    if mode =='test':
        log.write('test submission .... @ %s\n'%str(augment))
        csv_file = out_dir +'/submit/%s/resnet34-fpn256-tta-grid-search.csv'%(mode)


        predict_mask  = probability_mask>(np.array(threshold_pixel)*255).astype(np.uint8).reshape(1,4,1,1)
        predict_mask  = remove_small(predict_mask, threshold_size)
        predict_label = ((predict_mask.sum(-1).sum(-1))>0).astype(np.int32)


        image_id_class_id = []
        encoded_pixel = []
        for b in range(len(image_id)):
            for c in range(4):
                image_id_class_id.append(image_id[b]+'_%d'%(c+1))

                if predict_label[b,c]==0:
                    rle=''
                else:
                    rle = run_length_encode(predict_mask[b,c])
                encoded_pixel.append(rle)

        df = pd.DataFrame(zip(image_id_class_id, encoded_pixel), columns=['ImageId_ClassId', 'EncodedPixels'])
        df.to_csv(csv_file, index=False)

        ## print statistics ----
        text = print_submission_csv(df)
        log.write('\n')
        log.write('%s'%(text))



    exit(0)

'''
0 /root/share/project/kaggle/2019/steel/result5/resnet34-unet256-fold-0/submit/test
1 /root/share/project/kaggle/2019/steel/result5/resnet34-unet256-foldb0-0/submit/test
2 /root/share/project/kaggle/2019/steel/result5/resnet34-unet256-foldb1-0/submit/test
compare with LB probing ... 
		num_image =  1801(1801) 
		num  =  7204(7204) 
		neg  =  6371(6172)  0.884 
		pos  =   833(1032)  0.116 
		pos1 =   121( 128)  0.067  0.145 
		pos2 =    28(  43)  0.016  0.034 
		pos3 =   567( 741)  0.315  0.681 
		pos4 =   117( 120)  0.065  0.140 
 

##----

resnet34-unet256-foldb0-0 single **
submitting .... @ ['null', 'flip_lr', 'flip_ud']
threshold_pixel = [0.65, 0.65, 0.6, 0.4]
threshold_size  = [400, 600, 1000, 2500]

test submission .... @ ['null', 'flip_lr', 'flip_ud']

compare with LB probing ... 
		num_image =  1801(1801) 
		num  =  7204(7204) 
		neg  =  6330(6172)  0.879 
		pos  =   874(1032)  0.121 
		pos1 =   131( 128)  0.073  0.150 
		pos2 =    44(  43)  0.024  0.050 
		pos3 =   576( 741)  0.320  0.659 
		pos4 =   123( 120)  0.068  0.141 
 


'''


def run_ensemble():
    dir=[
        '/root/share/project/kaggle/2019/steel/result5/resnet34-unet256-fold-0/submit/test',
        '/root/share/project/kaggle/2019/steel/result5/resnet34-unet256-foldb0-0/submit/test',
        '/root/share/project/kaggle/2019/steel/result5/resnet34-unet256-foldb1-0/submit/test',
    ]

    mode = 'test'

    for t,d in enumerate(dir):
        print(t,d)
        id          = read_list_from_file(d +'/image_id.txt')
        truth       = np.load(d +'/truth_mask.uint8.npz')['arr_0']
        probability = np.load(d +'/probability_mask.uint8.npz')['arr_0']
        probability = probability.astype(np.float32) /255

        if t==0:
            image_id = id
            truth_mask = truth
            probability_mask = probability
        else:
            assert(image_id == id)
            probability_mask += probability

    probability_mask = probability_mask/len(dir)
    probability_mask = (probability_mask*255).astype(np.uint8)

    #decode
    num_test= len(image_id)
    value = np.max(probability_mask,1,keepdims=True)
    value = probability_mask*(value==probability_mask)
    probability_mask = probability_mask[:,1:] #remove background class

    index = np.ones((num_test,4,256,1600),np.uint8)*np.array([1,2,3,4],np.uint8).reshape(1,4,1,1)
    truth_mask = truth_mask==index

    #---

    threshold_pixel = [0.650,0.650,0.600,0.400,]
    threshold_size  = [400,600,1000,2500,]

    if mode =='test':
        out_dir  ='/root/share/project/kaggle/2019/steel/result5/resnet34-unet256-foldb0-0'
        csv_file = out_dir +'/resnet34-ensmble-x3.csv'


        predict_mask  = probability_mask>(np.array(threshold_pixel)*255).astype(np.uint8).reshape(1,4,1,1)
        predict_mask  = remove_small(predict_mask, threshold_size)
        predict_label = ((predict_mask.sum(-1).sum(-1))>0).astype(np.int32)

        image_id_class_id = []
        encoded_pixel = []
        for b in range(len(image_id)):
            for c in range(4):
                image_id_class_id.append(image_id[b]+'_%d'%(c+1))

                if predict_label[b,c]==0:
                    rle=''
                else:
                    rle = run_length_encode(predict_mask[b,c])
                encoded_pixel.append(rle)

        df = pd.DataFrame(zip(image_id_class_id, encoded_pixel), columns=['ImageId_ClassId', 'EncodedPixels'])
        df.to_csv(csv_file, index=False)

        ## print statistics ----
        text = print_submission_csv(df)
        print('%s'%(text))





# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    #run_submit_segmentation()
    run_ensemble()
