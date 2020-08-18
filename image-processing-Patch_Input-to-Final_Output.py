# ------------------------------------------------------------------------------
# Project name: 'Detecting Field Plots from Aerial Images of Wheat Fields'
# Input: High resolution Drone images of 2017 Crop breeding trials from P2IRC project
# Output: Downscaled images with detected field columns and field plots
# Steps:
#       - Downscale the TIFF images to PNG
#       - Denoise the PNG image
#       - Perform custom grayscale conversion
#       - Create image patches manually and keep them in 'patches' folder
#       - Take each patch and do the following:
#           - Detect field columns
#           - Expand each column individually
#           - Detect plots from each column
#           - Each each plot individually
#           - Generate column and row bins, and evaluate them against groud truth
# ------------------------------------------------------------------------------

# import necessary packages
import os as os
import numpy as np
import scipy.misc as m
import skimage.io as io
import skimage.util as util
import skimage.color as color
import skimage.filters as filt
import skimage.measure as meas
import matplotlib.pyplot as plt
import skimage.transform as sktr
import skimage.restoration as rest
import skimage.morphology as morph
import skimage.segmentation as seg
import matplotlib.patches as patches
import skimage.external.tifffile as tiff
import scipy.ndimage.morphology as morph2

# set image and ground truth paths
img_path = os.path.join('.', 'patches')
gt_path = os.path.join('.', 'groundtruth')

# Compute Dice coefficient
def DSC_measure(I, GT):
    im1 = np.asarray(I).astype(np.bool)
    im2 = np.asarray(GT).astype(np.bool)
    intersection = np.logical_and(im1, im2)
    return 2.0 * intersection.sum() / (im1.sum() + im2.sum())

dsc_array = []
rr_array = []
mr_array = []

# Need to count manually and insert these two array from ground truth and generated output image
region_count_gt = [19, 19, 19, 19, 19, 19, 38, 38, 38, 57, 57, 114]
misidentification_count_gt = [1, 1, 6, 4, 5, 5, 3, 9, 10, 9, 15, 21]


# START of the image processing steps
# taking all images as input from 'patches' folder

for root, dirs, files in os.walk(img_path):
    for filename in files:
        # ignore files that are not PNG files.
        if filename[-4:] != '.png':
            continue
        I = io.imread(os.path.join(img_path,filename))
        I = util.img_as_float(I)

        # gray conversion needed for patches
        I = color.rgb2gray(I)

        # OTSU Thresholding
        t = filt.threshold_otsu(I)
        bin = I > t

        # bin_otsu image pixels are scaled within 0 and 1
        bin_otsu = util.img_as_float(bin)

        # Save the result and uncomment to get the OTSU image

        # plt.imsave('sample-output/OTSU_' + filename, bin_otsu)
        # plt.imshow(bin_otsu)
        # plt.show()

        fieldcolumntopleftcolumn = []
        fieldcolumntopleftrow = []
        fieldcolumnbottomrightcolumn = []
        fieldcolumnbottomrightrow = []

        columnwhitepixelcount = [0, 0]
        firstwhitecolumnpixel = 0
        number_of_columns = 0
        rowtopflag = 0
        rowbottomflag = 0


        # Create a Rectangle patch
        row, column = bin_otsu.shape
        row_allowed_pixel_perc = 0.5

        # Calculation for column detection without expansion
        for j in range(column):
            for i in range(row):
                if(bin_otsu[i, j] == 1.0):
                    if(columnwhitepixelcount[1] == 0):
                        rowtopflag = i
                    rowbottomflag = i
                    columnwhitepixelcount[1] += 1

            if columnwhitepixelcount[1] >= row * row_allowed_pixel_perc and columnwhitepixelcount[0] < row * row_allowed_pixel_perc:
                #new column starting line found
                # print('new column start found')
                number_of_columns += 1
                fieldcolumntopleftcolumn.append(j)
                fieldcolumntopleftrow.append(rowtopflag)

            if columnwhitepixelcount[1] < row * row_allowed_pixel_perc and columnwhitepixelcount[0] >= row * row_allowed_pixel_perc:
                # column end line found
                # print('new column end found')
                fieldcolumnbottomrightcolumn.append(j-1)
                fieldcolumnbottomrightrow.append(rowbottomflag)

            columnwhitepixelcount[0] = columnwhitepixelcount[1]
            columnwhitepixelcount[1] = 0

        fig, ax = plt.subplots(1)

        # Uncomment to see the Output for Column Detection without expansion
        # for i in range(number_of_columns):
        #    rectwidth = fieldcolumnbottomrightcolumn[i] - fieldcolumntopleftcolumn[i]
        #    rectheight = fieldcolumnbottomrightrow[i] - fieldcolumntopleftrow[i]
        #    rect = patches.Rectangle((fieldcolumntopleftcolumn[i], fieldcolumntopleftrow[i]), rectwidth, rectheight,
        #                             linewidth=1, edgecolor='green', facecolor='none')
        #    ax.add_patch(rect)


        # outputfile= 'sample-output/column_detection_' + filename
        # plt.imshow(bin_otsu)
        # plt.savefig(outputfile)
        # plt.show()
        # End of Output for Column Detection


        field_column_allowed_pixel_perc = 0.2

        # Calculation for expansion of columns towards top and bottom
        for i in range(len(fieldcolumntopleftcolumn)):
            #print("Column : " + str(i))
            row_index = fieldcolumntopleftrow[i]

            column_width = fieldcolumnbottomrightcolumn[i] - fieldcolumntopleftcolumn[i]

            loop_break_flag = True

            while loop_break_flag:
                white_pixel_count = 0
                for j in range(fieldcolumntopleftcolumn[i], fieldcolumnbottomrightcolumn[i]):
                    if (bin_otsu[row_index, j] == 1.0):
                        white_pixel_count += 1

                if(white_pixel_count >= column_width * field_column_allowed_pixel_perc):
                    row_index -= 1
                else:
                    fieldcolumntopleftrow[i] = row_index
                    loop_break_flag = False

            loop_break_flag = True
            row_index = fieldcolumnbottomrightrow[i]
            while loop_break_flag:
                white_pixel_count = 0
                for j in range(fieldcolumntopleftcolumn[i], fieldcolumnbottomrightcolumn[i]):
                    if (bin_otsu[row_index, j] == 1.0):
                        white_pixel_count += 1

                if(white_pixel_count >= column_width * field_column_allowed_pixel_perc):
                    row_index += 1
                else:
                    fieldcolumnbottomrightrow[i] = row_index
                    loop_break_flag = False


        # Uncomment to get the image with column detection with expansion
        # fig, ax = plt.subplots(1)
        # ax.imshow(bin_otsu)


        # for i in range(number_of_columns):
        #    rectwidth = fieldcolumnbottomrightcolumn[i] - fieldcolumntopleftcolumn[i]
        #    rectheight = fieldcolumnbottomrightrow[i] - fieldcolumntopleftrow[i]
        #    rect = patches.Rectangle((fieldcolumntopleftcolumn[i], fieldcolumntopleftrow[i]), rectwidth, rectheight,
        #                             linewidth=1, edgecolor='cyan', facecolor='none')
        #
        #     # Add the patch to the Axes
        #    ax.add_patch(rect)

        # outputfile = 'sample-output/column_expansion_' + filename
        # plt.imshow(bin_otsu)
        # plt.savefig(outputfile)
        # plt.show()



        ## PLOT Segmentation
        #fig, ax = plt.subplots(1)

        # Display the image
        #ax.imshow(bin_otsu)

        # Create a Rectangle patch for plots

        fieldplottopleftcolumn = []
        fieldplottopleftrow = []
        fieldplotbottomrightcolumn = []
        fieldplotbottomrightrow = []
        rowwhitepixelcount=[0,0]
        numberofrows = []

        #print('Number of rows:\n')
        #print(numberofrows)

        from_range = 0
        to_range = 0

        # Calculation for plot detection inside column
        for k in range(0, number_of_columns):
            # k=int(k/2)
            numberofrows.append(0)
            for i in range(fieldcolumntopleftrow[k],fieldcolumnbottomrightrow[k]+1):
                rowwidth = fieldcolumnbottomrightrow[k] - fieldcolumntopleftrow[k]
                columnwidth = fieldcolumnbottomrightcolumn[k] - fieldcolumntopleftcolumn[k]
                for j in range(fieldcolumntopleftcolumn[k],fieldcolumnbottomrightcolumn[k]):
                    if (bin_otsu[i, j] == 1.0):
                       rowwhitepixelcount[1] += 1

                if (rowwhitepixelcount[1] >= columnwidth*.2 and rowwhitepixelcount[0] < columnwidth * field_column_allowed_pixel_perc ):
                    # new plot start found
                    numberofrows[k]+=1
                    fieldplottopleftcolumn.append(fieldcolumntopleftcolumn[k])
                    fieldplottopleftrow.append(i)
                if (rowwhitepixelcount[1] < columnwidth*.2 and rowwhitepixelcount[0] >= columnwidth * field_column_allowed_pixel_perc):
                    # plot end found
                    fieldplotbottomrightcolumn.append(fieldcolumnbottomrightcolumn[k])
                    fieldplotbottomrightrow.append(i)

                rowwhitepixelcount[0]=rowwhitepixelcount[1];
                rowwhitepixelcount[1]=0;


            min_column = 10000000000
            max_column = -10000000000

            # calculation for plot expansion and also find minimum from
            # left side and maximum from right side

            to_range = from_range + numberofrows[k]

            for i in range(from_range, to_range):

                column_index = fieldplottopleftcolumn[i]
                row_height = fieldplotbottomrightrow[i] - fieldplottopleftrow[i]

                loop_break_flag = True

                while loop_break_flag:
                    white_pixel_count = 0
                    for j in range(fieldplottopleftrow[i], fieldplotbottomrightrow[i]):
                        if (bin_otsu[j, column_index] == 1.0):
                            white_pixel_count += 1

                    if (white_pixel_count >= row_height * field_column_allowed_pixel_perc):
                        column_index -= 1
                    else:
                        fieldplottopleftcolumn[i] = column_index
                        loop_break_flag = False
                        if column_index < min_column:
                            min_column = column_index

                column_index = fieldplotbottomrightcolumn[i]
                row_height = fieldplotbottomrightrow[i] - fieldplottopleftrow[i]

                loop_break_flag = True

                while loop_break_flag:
                    white_pixel_count = 0
                    for j in range(fieldplottopleftrow[i], fieldplotbottomrightrow[i]):
                        if (bin_otsu[j, column_index] == 1.0):
                            white_pixel_count += 1

                    if (white_pixel_count >= row_height * field_column_allowed_pixel_perc):
                        column_index += 1
                    else:
                        fieldplotbottomrightcolumn[i] = column_index
                        loop_break_flag = False
                        if column_index > max_column:
                            max_column = column_index


            # Setting minimum for left side and maximum for right side
            fieldcolumntopleftcolumn[k] = min_column
            fieldcolumnbottomrightcolumn[k] = max_column

            rectwidth = fieldcolumnbottomrightcolumn[k] - fieldcolumntopleftcolumn[k]
            rectheight = fieldcolumnbottomrightrow[k] - fieldcolumntopleftrow[k]
            rect = patches.Rectangle((fieldcolumntopleftcolumn[k], fieldcolumntopleftrow[k]), rectwidth, rectheight,
                                     linewidth=1, edgecolor='cyan', facecolor='none')
            ax.add_patch(rect)

            # Add the patch to the Axes
            # Draw red rectangles for plot detection and final outout
            for i in range(from_range, to_range):
                rectwidth = fieldplotbottomrightcolumn[i] - fieldplottopleftcolumn[i]
                rectheight = fieldplotbottomrightrow[i] - fieldplottopleftrow[i]

                rect = patches.Rectangle((fieldplottopleftcolumn[i], fieldplottopleftrow[i]), rectwidth,
                                         rectheight, linewidth=1, edgecolor='red', facecolor='none')
                ax.add_patch(rect)

            from_range += numberofrows[k]

        outputfile = 'results/output/final_output_' + filename
        plt.imshow(bin_otsu)
        plt.savefig(outputfile)
        #plt.show()

        fig1 = plt.figure()
        img_binary_output = np.zeros_like(I)

        #print(fieldcolumntopleftrow)
        # Calculation and image generation for binary image of column detection
        for column_no_index in range(number_of_columns):
            # print(column_index)
            for row_index in range(fieldcolumntopleftrow[column_no_index], fieldcolumnbottomrightrow[column_no_index]):
                for column_index in range(fieldcolumntopleftcolumn[column_no_index], fieldcolumnbottomrightcolumn[column_no_index]):
                    img_binary_output[row_index, column_index] = 1


        outputfile = 'results/binary/final_output_bin_column_' + filename
        #outputfile = 'sample-output/final_output_bin_column_' + filename
        plt.imshow(img_binary_output)
        plt.savefig(outputfile)
        #plt.show()

        #print(number_of_columns)
        #print(numberofrows)

        # Calculation and image generation for binary image of plot detection
        # This image will be compared with ground truth
        img_binary_output = np.zeros_like(I)
        for row_no_index in range(0, len(fieldplottopleftrow)):
            # print(column_index)
            for row_index in range(fieldplottopleftrow[row_no_index], fieldplotbottomrightrow[row_no_index]):
                for column_index in range(fieldplottopleftcolumn[row_no_index], fieldplotbottomrightcolumn[row_no_index]):
                    img_binary_output[row_index, column_index] = 1


        outputfile = 'results/binary/final_output_bin_plot_' + filename
        plt.imshow(img_binary_output)
        plt.savefig(outputfile)
        #plt.show()

        plt.close('all')

        print("File Name = " + filename)
        print("Number of column = " + str(number_of_columns))
        print("Number of rows per column = " + str(numberofrows))

        ##### GT Load and processing #############
        # Load Ground Truth image
        I_GT = io.imread(os.path.join(gt_path, filename.replace(".","_GT.")))
        I_GT = util.img_as_float(I_GT)

        # gray conversion needed for patches
        I_GT = color.rgb2gray(I_GT)
        t = filt.threshold_otsu(I_GT)
        bin_gt = I_GT > t

        # Calculate DSC values
        DSC_val = DSC_measure(img_binary_output, bin_gt)
        dsc_array.append(DSC_val)
        print("DSC Measurement = " + str(DSC_val))

        img_index = int(filename[:-4]) - 1

        gt_region_count = region_count_gt[img_index]
        image_region_count = sum(numberofrows)

        # Calculate recognition rate, RR
        RR = image_region_count / gt_region_count
        rr_array.append(RR)
        print("Recognised Rate = " + str(RR))

        # Calculate mis-identification rate, MR
        gt_misidentification_count = misidentification_count_gt[img_index]
        MR = gt_misidentification_count / gt_region_count
        mr_array.append(MR)

        print("Misidentification Rate = " + str(MR))
        print("--------------------------------------------------------")

    print("--------------------------------------------------------")
    avg_DSC = sum(dsc_array) / len(dsc_array)
    avg_RR = sum(rr_array) / len(rr_array)
    avg_MR = sum(mr_array) / len(mr_array)

    # Calculate the averages of DSC, RR and MR
    print("Average measurement :")
    print("Average DSC = " + str(avg_DSC))
    print("Average RR = " + str(avg_RR))
    print("Average MR = " + str(avg_MR))
    print("End of code !!")
