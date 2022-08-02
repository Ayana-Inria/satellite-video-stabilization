import cv2
import numpy as np
import os 
import csv

def read_csv(directory):
    '''
        Returns a np array of size [N_frames x N_objects * 6], each object has (px, py, vx, vy, ax, ay) components
    '''

    objects_list = []
    with open(directory) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        frame_counter = 0
        for row in spamreader:
            frame_counter += 1
            if(frame_counter == 1):
                continue
            objects_list.append(row)
    objects_array = np.array(objects_list)
    return objects_array


def save_AOIs(directory, im_name, transformed_img, resolution, output_folder, downsample=0):
    im_name = im_name.split('/')[-1]
    if not os.path.exists(directory + '/' + output_folder):
      os.makedirs(directory + '/' + output_folder)

    if not os.path.exists(directory + '/AOIs/AOI_01/' + output_folder):
        os.makedirs(directory + '/AOIs/AOI_01/' + output_folder)

    if not os.path.exists(directory + '/AOIs/AOI_02/' + output_folder):
        os.makedirs(directory + '/AOIs/AOI_02/' + output_folder)

    if not os.path.exists(directory + '/AOIs/AOI_03/' + output_folder):
        os.makedirs(directory + '/AOIs/AOI_03/' + output_folder)

    if not os.path.exists(directory + '/AOIs/AOI_04/' + output_folder):
        os.makedirs(directory + '/AOIs/AOI_04/' + output_folder)

    if not os.path.exists(directory + '/AOIs/AOI_34/' + output_folder):
        os.makedirs(directory + '/AOIs/AOI_34/' + output_folder)

    if not os.path.exists(directory + '/AOIs/AOI_40/' + output_folder):
        os.makedirs(directory + '/AOIs/AOI_40/' + output_folder)


    if not os.path.exists(directory + '/AOIs/AOI_41/' + output_folder):
        os.makedirs(directory + '/AOIs/AOI_41/' + output_folder)

    if not os.path.exists(directory + '/AOIs/AOI_42/' + output_folder):
        os.makedirs(directory + '/AOIs/AOI_42/' + output_folder)


    # yo = 8700; xo = 8390; im = (frame_000100(xo:xo + 2278, yo: yo + 2278)); imshow(im);
    if(resolution == 0):
      # Area 01
      yo_01 = 7373 // downsample
      xo_01 = 14531 // downsample
      yf_01 =  9701// downsample
      xf_01 = 16687 // downsample
      # cv2.imwrite(directory + '/AOIs/AOI_02/gt/' + im_name, transformed_gt[xo: xo + 2278, yo: yo + 2278])


      # Area 02
      yo_02 = 8700 // downsample
      xo_02 = 8390 // downsample
      xf_02 = xo_02 + 2278 // downsample
      yf_02 =  yo_02 + 2278 // downsample
      # cv2.imwrite(directory + '/AOIs/AOI_02/gt/' + im_name, transformed_gt[xo: xo + 2278, yo: yo + 2278])

      # Area 34
      yo_34 = 7253 // downsample
      xo_34 = 9750 // downsample
      xf_34 = xo_34 + 2660 // downsample
      yf_34 = yo_34 + 4260 // downsample
      # cv2.imwrite(directory + '/AOIs/AOI_34/gt/' + im_name, transformed_gt[xo: xo + 2660, yo: yo + 4260])

      # Area 40
      yo_40 = 6700 // downsample
      xo_40 = 13900 // downsample
      yf_40 = yo_40 + 3265 // downsample
      xf_40 = xo_40 + 2542  // downsample
      # cv2.imwrite(directory + '/AOIs/AOI_40/gt/' + im_name, transformed_gt[xo: xo + 2542, yo: yo + 3265])


      # Area 41
      yo_41 = 8890 // downsample
      xo_41 = 13960 // downsample
      yf_41 = yo_41 + 3207 // downsample
      xf_41 = xo_41 + 2892 // downsample
    else:
      # resolution = 2

      # Area 1
      yo_01 = 1842 // downsample
      xo_01 = 3610 // downsample
      yf_01 = 2400 // downsample
      xf_01 = 4189 // downsample

      # Area 2
      yo_02 = 2175 // downsample
      xo_02 = 2189 // downsample
      xf_02 = 2685 // downsample
      yf_02 = 2760 // downsample


      # Area 3
      yo_03 = 4162 // downsample
      xo_03 = 2605 // downsample
      yf_03 = 4752 // downsample
      xf_03 = 3130 // downsample


      # Area 4
      yo_04 = 2195 // downsample
      xo_04 = 4100 // downsample
      yf_04 = 2756 // downsample
      xf_04 = 4653 // downsample

      # Area 34
      yo_34 = 1746 // downsample
      xo_34 = 2441 // downsample
      yf_34 = 2899 // downsample
      xf_34 = 3097 // downsample
      # cv2.imwrite(directory + '/AOIs/AOI_34/gt/' + im_name, transformed_gt[xo: xo + 2660, yo: yo + 4260])

      # Area 40
      yo_40 = 1730 // downsample
      xo_40 = 3499 // downsample
      yf_40 = 2545 // downsample
      xf_40 = 4112 // downsample
      # cv2.imwrite(directory + '/AOIs/AOI_40/gt/' + im_name, transformed_gt[xo: xo + 2542, yo: yo + 3265])

      # Area 41
      yo_41 = 2085 // downsample
      xo_41 = 3463 // downsample
      yf_41 = 3061 // downsample
      xf_41 = 4232 // downsample


      # Area 42
      yo_42 = 2198 // downsample
      xo_42 = 3474 // downsample
      yf_42 = 2671 // downsample
      xf_42 = 3842 // downsample

    # print(directory + '/AOIs/AOI_01/' + output_folder + '/' + im_name)
    # exit()
    cv2.imwrite(directory + '/AOIs/AOI_01/' + output_folder + '/' + im_name, transformed_img[xo_01:xf_01, yo_01:yf_01])
    cv2.imwrite(directory + '/AOIs/AOI_02/' + output_folder + '/' + im_name, transformed_img[xo_02:xf_02, yo_02:yf_02])
    cv2.imwrite(directory + '/AOIs/AOI_34/' + output_folder + '/' + im_name, transformed_img[xo_34:xf_34, yo_34:yf_34])
    cv2.imwrite(directory + '/AOIs/AOI_40/' + output_folder + '/' + im_name, transformed_img[xo_40:xf_40, yo_40:yf_40])
    cv2.imwrite(directory + '/AOIs/AOI_41/' + output_folder + '/' + im_name, transformed_img[xo_41:xf_41, yo_41:yf_41])

    if(resolution > 0):
        cv2.imwrite(directory + '/AOIs/AOI_03/' + output_folder + '/' + im_name, transformed_img[xo_03:xf_03, yo_03:yf_03])
        cv2.imwrite(directory + '/AOIs/AOI_04/' + output_folder + '/' + im_name, transformed_img[xo_04:xf_04, yo_04:yf_04])
        cv2.imwrite(directory + '/AOIs/AOI_42/' + output_folder + '/' + im_name, transformed_img[xo_42:xf_42, yo_42:yf_42])
    
    # Save the output.
    print("Writting at")
    print(directory + '/' + output_folder + "/" + im_name)
    cv2.imwrite(directory + '/' + output_folder + "/" + im_name, transformed_img)
    

def match_two_images(input_directory, im_ref_path, transformed_image_path, output_directory, resolution=2):
    frame_number = int(transformed_image_path[-10:-4])
    print("reference image: " + im_ref_path)
    print("transforming image: " + transformed_image_path)

    # Open the image files.
    reference_image_color = cv2.imread(input_directory + "/" + im_ref_path)
    trgt_img_color = cv2.imread(input_directory + "/" +  transformed_image_path)
    # Convert to grayscale.
    trgt_img = cv2.cvtColor(trgt_img_color, cv2.COLOR_BGR2GRAY)
    try:
        ref_img = cv2.cvtColor(reference_image_color, cv2.COLOR_BGR2GRAY)
    except:
        ref_img = reference_image_color

    height, width = ref_img.shape
     
    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(50000)
     
    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not required in this case).
    trgt_img_mask = 1 - (trgt_img == 0).astype(np.uint8) - (trgt_img == 255).astype(np.uint8)
    ref_img_mask = 1 - (ref_img == 0).astype(np.uint8) - (ref_img == 255).astype(np.uint8)
    kp1, d1 = orb_detector.detectAndCompute(trgt_img, mask=trgt_img_mask)
    kp2, d2 = orb_detector.detectAndCompute(ref_img, mask=ref_img_mask)
     
    # Match features between the two images.
    # We create a Brute Force matcher with
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
     
    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)

    # Sort matches on the basis of their Hamming distance.
    # matches.sort(key = lambda x: x.distance)
    matches = sorted(matches, key = lambda x:x.distance)
     
    # Take the top 80 % matches forward.
    matches = matches[:int(len(matches)*0.8)]
    no_of_matches = len(matches)
     
    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
      p1[i, :] = kp1[matches[i].queryIdx].pt
      p2[i, :] = kp2[matches[i].trainIdx].pt
     
    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    if not os.path.exists(output_directory + '/homography_transforms'):
      os.makedirs(output_directory + '/homography_transforms')

    np.save(output_directory + "/homography_transforms/" + transformed_image_path[-10:-4], homography)
    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(trgt_img, homography, (width, height))       
    
    save_AOIs(output_directory, transformed_image_path, transformed_img, resolution, output_folder="images", downsample=1)

    return


def AOI_gt_csv_generator(input_directory, output_directory, resolution):
    # [row, column]
    if resolution == 0:
        AOI_numbers = [1, 2, 34, 40, 41]
        AOI_start_coords = [
                            [14531, 7373],
                            [8390, 8700],
                            [9750, 7253],
                            [13900, 6700],
                            [13960, 8890],
                            ]

        AOI_end_coords = [
                          [16687, 9701],
                          [8390 + 2278, 8700 + 2278],
                          [9750 + 2660, 7253 + 4260],
                          [13900 + 2542, 6700 + 3265],
                          [13960 + 2892, 8890 + 3207],
                        ]
    if resolution == 2:
        AOI_numbers = [1, 2, 3, 4, 34, 40, 41, 42]
                            # x,     y
        AOI_start_coords = [
                            [3610, 1842],
                            [2189, 2175],
                            [2605, 4162],
                            [4100, 2195],
                            [2441, 1746],
                            [3499, 1730],
                            [3463, 2085],
                            [3474, 2198]
                            ]

        AOI_end_coords = [
                          [4189, 2400],
                          [2685, 2760],
                          [3130, 4752],
                          [4653, 2756],
                          [3097, 2899],
                          [4112, 2545],
                          [4232, 3061],
                          [3842, 2671]
                        ]

    num_aois = len(AOI_numbers)

    for i in range(num_aois):
        number = AOI_numbers[i]
        start_coords = AOI_start_coords[i]
        end_coords = AOI_end_coords[i]

        print("Processing for AOI {}".format(number))
        transform_gt_to_csv_for_AOIS(input_directory, output_directory, resolution, number, start_coords, end_coords)


def transform_gt_to_csv_for_AOIS(input_directory, output_directory, resolution, AOI_number, AOI_start_coords, AOI_end_coords):
    '''
        Attention: x coordinate refers to cols
                   y coordinate refers to rows   
    '''
    # directory = 'r' + str(resolution)
    output_directory_images = output_directory
    csv_file_dir = input_directory + "TrackTruth/TRAIN/20091021_truth_rset" + str(resolution) + "_frames0100-0611.csv"

    if not os.path.exists(output_directory + '/AOIs/AOI_' + str(AOI_number).zfill(2) + '/measurements'):
        os.makedirs(output_directory + '/AOIs/AOI_' + str(AOI_number).zfill(2) + '/measurements')

    # Get the stabilization transforms
    transforms_dict = {}
    transform_files = sorted([f for f in os.listdir(output_directory + '/homography_transforms') if os.path.isfile(os.path.join(output_directory + '/homography_transforms', f))])
    for file in transform_files:
        number = int(file[:-4])
        transforms_dict[number] = np.load(output_directory + '/homography_transforms/' + file)
    print("Reading GT csv file for formatting")
    object_array = read_csv(csv_file_dir)

    # Find how many labels I have
    label_dict = {}
    label_counter = 1

    for gt_row in object_array:
        label = int(gt_row[0])
        i = int(gt_row[3])
        j = int(gt_row[4])
        gt_frame = int(gt_row[6])
        
        #         print(gt_frame)
        #        exit()
        if(gt_frame > 100):
            if gt_frame > 110:
                gt_frame_temp = 110
            else:
                gt_frame_temp = gt_frame
            transform = transforms_dict[gt_frame_temp]
            src_pts = np.array([[i, j], ]).reshape(-1,1,2).astype(np.float32)
            pts = cv2.perspectiveTransform(src_pts, transform).astype(np.int32)
            new_i = pts[0, 0, 0]
            new_j = pts[0, 0, 1]
        else:
            new_i = i
            new_j = j

        if(new_i >= AOI_start_coords[1] and new_i < AOI_end_coords[1] and new_j >= AOI_start_coords[0] and new_j < AOI_end_coords[0]):
            # In these annotations from the US Airforce, the column coordinate comes first, so I interchange j and i.
            if(label in label_dict.keys()):
                continue
            else:
                label_dict[label] = label_counter
                label_counter += 1

    # Objective: I want 6 coordinates px, py, vx, vy, ax, ay
    print("Found {} labels".format(label_counter))
    # labels = 22981
    transformed_gt = np.zeros((512 + 1 , 1 + (label_counter * 6)))
    new_array = []

    # Read transforms
    list_of_transforms = []

    image111 = cv2.imread(output_directory_images + '/AOIs/AOI_' + str(AOI_number).zfill(2) + '/images/frame_000104.png')
    # image112 = cv2.imread(output_directory_images + '/AOIs/AOI_' + str(AOI_number).zfill(2) + '/images/frame_000112.png')
    # image400 = cv2.imread(output_directory_images + '/AOIs/AOI_' + str(AOI_number).zfill(2) + '/images/frame_000400.png')

    rows, cols, _ = image111.shape
    measurement_frames = np.zeros((rows, cols, 512), dtype=np.uint16)
    for gt_row in object_array:
        label = int(gt_row[0])
        i = int(gt_row[3])
        j = int(gt_row[4])
        gt_frame = int(gt_row[6])
        
        if(gt_frame > 100):
            if gt_frame > 110:
                gt_frame_temp = 110
            else:
                gt_frame_temp = gt_frame
            transform = transforms_dict[gt_frame_temp]

            src_pts = np.array([[i, j], ]).reshape(-1,1,2).astype(np.float32)
            pts = cv2.perspectiveTransform(src_pts, transform).astype(np.int32)
            new_i = pts[0, 0, 0]
            new_j = pts[0, 0, 1]
        else:
            new_i = i
            new_j = j

        if(new_i >= AOI_start_coords[1] and new_i < AOI_end_coords[1] and new_j >= AOI_start_coords[0] and new_j < AOI_end_coords[0]):
            # In these annotations from the US Airforce, the column coordinate comes first, so I interchange j and i.
            indx = label_dict[label]

            shifted_j = new_j - AOI_start_coords[0]
            shifted_i = new_i - AOI_start_coords[1]

            transformed_gt[gt_frame - 100, (indx - 1) * 6 + 1] = shifted_j 
            transformed_gt[gt_frame - 100, (indx - 1) * 6 + 2] = shifted_i
            transformed_gt[gt_frame - 100, 0] = gt_frame

            if(gt_frame == 104):
                image111 = cv2.circle(image111, (shifted_i, shifted_j), 4, (0, 0, 255), 1)

            '''
            if(gt_frame == 112):
                image112 = cv2.circle(image112, (shifted_i, shifted_j), 4, (0, 0, 255), 1)

            if(gt_frame == 400):
                image400 = cv2.circle(image400, (shifted_i, shifted_j), 4, (0, 0, 255), 1)
            '''
            measurement_frames[shifted_j, shifted_i, gt_frame - 100] = label

    # Write Images for Assurance
    cv2.imwrite(output_directory_images + '/AOIs/AOI_' + str(AOI_number).zfill(2) + '/debug_image111.png', image111)
    # cv2.imwrite(output_directory_images + '/AOIs/AOI_' + str(AOI_number).zfill(2) + '/debug_image112.png', image112)
    # cv2.imwrite(output_directory_images + '/AOIs/AOI_' + str(AOI_number).zfill(2) + '/debug_image400.png', image400)

    # Write Images for debugging
    for k in range(measurement_frames.shape[-1]):
        cv2.imwrite(output_directory_images + '/AOIs/AOI_' + str(AOI_number).zfill(2) + '/measurements/measurement_' + str(k + 100).zfill(4) + ".png", measurement_frames[:, :, k])

    # Write everything 
    output_csv_path =  output_directory_images + "/AOIs/AOI_" + str(AOI_number).zfill(2) + "/transformed_object_states.csv"
    with open(output_csv_path,  mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header_content = []
        header_content.append('Frame Number')
        for object_n in label_dict.keys():
            header_content.append('obj_{}_px'.format(object_n))
            header_content.append('obj_{}_py'.format(object_n))
            header_content.append('obj_{}_vx'.format(object_n))
            header_content.append('obj_{}_vy'.format(object_n))
            header_content.append('obj_{}_ax'.format(object_n))
            header_content.append('obj_{}_ay'.format(object_n))
        csv_writer.writerow(header_content)

        for row in transformed_gt:
            csv_writer.writerow(row)


def stabilize_images(input_directory, output_directory, resolution):
  input_directory_images = input_directory + "/r{}".format(resolution)
  # Coordinaes made only for r0 and r2
  resolution = 0
  file_names = sorted([f for f in os.listdir(input_directory_images + '/train_all') if os.path.isfile(os.path.join(input_directory_images + '/train_all', f))])
  ref_file = "frame_000100.png"
  old_file_name = ref_file
  counter = 0
  for file_n in file_names:
    if(counter == 0):
        temp_ref_file = "train_all/" + ref_file
    else:
        temp_ref_file = "images/" + ref_file

    match_two_images(input_directory_images, temp_ref_file, "train_all/" +  file_n, output_directory, resolution=resolution)
    if((counter + 1) % 30 == 0):
        ref_file = file_n
    counter += 1

def main():
    resolution = 0
    input_directory_images = "C:/Users/caguilar/Desktop/aguilar-camilo/Codes/Data_Processing/US_Army/"
    output_directory = "r{}/transformed_data".format(resolution)

    # stabilize_images(input_directory_images, output_directory, resolution)
    AOI_gt_csv_generator(input_directory_images, output_directory, resolution)

if __name__ == '__main__':
    main()