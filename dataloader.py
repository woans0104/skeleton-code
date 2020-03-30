from torch.utils.data.dataset import Dataset
import numpy as np
import torchvision.transforms as transforms
import random
import glob
import os
import cv2
import nibabel as nib
import ipdb
from PIL import Image
#from scipy.misc import imread, imresize,imsave





class CustomDataset(Dataset):

    def __init__(self, image_paths,target_paths,transform1,arg_mode=False,arg_thres= 0):   # initial logic happens like transform


        self.image_paths_pre ,self.target_paths_pre  = np.array(image_paths), np.array(target_paths)
        self.transforms1 = transform1
        self.arg_mode =arg_mode
        self.arg_thres = arg_thres




    def arg(self,image,mask,arg_thres):
        """
        # Resize

        resize = transforms.Resize(size=(256, 256))
        image = resize(image)
        mask = resize(mask)

        #from PIL import Image
        #image = Image.fromarray(image)
        #print(type(image))

        # Random horizontal flipping
        horizontal=random.random()
        #print('horizontal flipping', horizontal)
        if horizontal > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        
        # Random vertical flipping
        vertical=random.random()
        #print('vertical flipping', vertical)
        if vertical > 0.5:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)
        
        # Random rotation
        Rrotation = random.random()
        #print('Rrotation', Rrotation)
        if Rrotation > 0.5:
            angle = random.randint(-30, 30)
            image = transforms.functional.rotate(image,angle)
            mask = transforms.functional.rotate(mask,angle)


        """
        brightness = random.random()
        contrast = random.random()

        if brightness >arg_thres and contrast>arg_thres :

            #brightness_factor = random.uniform(0.3, 1.1)
            brightness_factor = random.uniform(0.5, 1.0)
            #brightness_factor = random.uniform(0.7, 1.1)
            image = transforms.functional.adjust_brightness(image, brightness_factor)

            #contrast_factor = random.uniform(0.3, 1.1)
            contrast_factor = random.uniform(0.3, 0.8)
            image = transforms.functional.adjust_contrast(image, contrast_factor)


        else:
            if brightness > arg_thres:
                brightness_factor = random.uniform(0.3,1.1)
                #brightness_factor = random.uniform(0.7, 1.1)
                image = transforms.functional.adjust_brightness(image, brightness_factor)


            if contrast > arg_thres:

                contrast_factor = random.uniform(0.3,0.8)
                #contrast_factor = random.uniform(0.3,1.1)
                #contrast_factor = random.uniform(0.7, 1.0)
                image = transforms.functional.adjust_contrast(image, contrast_factor)




        image = np.array(image)
        mask = np.array(mask)


        return image,mask


    def __getitem__(self, index):

        # indexing test

        image = self.image_paths_pre[index]
        mask = self.target_paths_pre[index]

        #image pre-processing

        #image = (image / 256) #16->8
        image = 255 * ((image - image.min()) / (image.max() - image.min())) #0~255 mapping
        image = image.astype('uint8')

        # find gm
        mask[mask != 1.0] = 0
        mask = mask.astype('uint8')



        #center_crop
        image_crop ,mask_crop = self.center_crop(image,mask,self.site)

    

        #arg
        if self.arg_mode:
            image_arg,mask= self.arg(image_crop,mask_crop,self.arg_thres)
            t_image = self.transforms1(image_arg)
        else:
            t_image = self.transforms1(image_crop)






       # mask = np.expand_dims(mask, -1)
        mask = np.array(mask_crop)
        mask[mask > 0] = 1
        mask = mask*255

        if len(set(mask.flatten())) != 2:
            print('here')
            import ipdb
            ipdb.set_trace()

      
        totensor = transforms.ToTensor()
        t_mask=totensor(mask)

        assert len(set(np.array(t_mask).reshape(-1))) ==2 ; 'error mask : ' + str(set(np.array(t_mask).reshape(-1)))


        return t_image, t_mask,np.array(image_crop)

    def __len__(self):  # return count of sample we have

        return len(self.image_paths_pre)




def make_dataset(site_name,train_size,mode='train'):

   


    # simple division : train_size_6,val_size_2,test_size_2
    # if valsize == false : train_size_7, test_size_3
    print('now dataset',site_name)

    valid_exts = ['.gz']  # 이 확장자들만 불러오겠다.

    if mode == 'train':
        image_folder = sorted(glob.glob(
            ('/data2/woans0104/Spinal_cord_dataset/training-data-gm-sc-challenge-ismrm16-v20160302b/{}-*'.format(site_name))))

        mask_folder = sorted(glob.glob(
            ('/data2/woans0104/Spinal_cord_dataset/voting_mask/{}-*'.format(
                site_name))))
    else:

        # if mode == 'test'
        train_size = 1
        image_folder = sorted(glob.glob(
            ('/data2/woans0104/Spinal_cord_dataset/test-data-gm-sc-challenge-ismrm16-v20160401/{}-*'.format(site_name))))


    # 폴더 안의 확장자들만 가져오기

    image_paths = []
    target_paths = mask_folder

    for f in image_folder:
        ext = os.path.splitext(f)[1]  # 확장자만 가져오기

        if ext.lower() not in valid_exts:
            continue
        if 'mask' not in os.path.splitext(f)[0].split('\\')[-1]:
            image_paths.append(f)


    print(len(image_paths))

    print(len(target_paths))

    assert len(image_paths) == len(target_paths); 'not same image & target lenths : image len : {} , target len : {}'.format(str(len(image_paths)),str(len(target_paths)))



    # last sort
    image_paths = sorted(image_paths)
    target_paths = sorted(target_paths)

   
    #split train/test

    len_data = len(image_paths)
    indices_image = list(range(len_data))


    #np.random.seed(random_seed)
    np.random.shuffle(indices_image)

    image_paths = np.array(image_paths)
    target_paths = np.array(target_paths)

    train_image_no = indices_image[:int(len_data * train_size)]
    test_image_no = indices_image[int(len_data * train_size):]

    train_image_paths = image_paths[train_image_no]
    train_mask_paths = target_paths[train_image_no]

    test_image_paths = image_paths[test_image_no]
    test_mask_paths = target_paths[test_image_no]

    print('end load data')



    return [train_image_paths, train_mask_paths],[test_image_paths, test_mask_paths]



