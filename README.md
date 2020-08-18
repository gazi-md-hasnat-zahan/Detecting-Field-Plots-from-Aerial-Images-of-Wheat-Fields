# Detecting-Field-Plots-from-Aerial-Images-of-Wheat-Fields

This project is a small contribution to the P2IRC project that is a seven-year innovative research and training program, Designing Crops for Global Food Security, will transform 
crop breeding and provide innovative solutions to national and global food security. The program builds upon the Global Institute for Food Securitys (GIFS) focus on computational 
agriculture to enhance the University of Saskatchewan biosciences cluster - one of the largest clusters of food-related researchers in the world [1].

## Objective
The purpose of this project was to detect the individual field plots from an aerial (drone) image of wheat fields. The image consisted of long field columns and several plots 
in each column. But there were other areas out of concern like- half of the field where the crops have been harvested already i.e. an open field. The other half that had plots 
was the actual part of the image that we worker on. So, the plan was to preprocess the image first into smaller patches having the area of interest and then go for the detection 
and further post-processing.

## Contribution
We started with some pre-processing of the aerial TIFF image using downscaling, noise reduction, converting to gray image and
cutting the image into several patches respectively. Then we used the patches to feed into our main algorithm
to find out the field columns first. After detecting the field columns, we detected the individual plots from
the columns and got the final output. As post-processing, we saved the converted the detected plots into
binary images and saved them for evaluation.

For evaluation, we did not have any ground truth from the data source. So we prepared the ground truths
using Adobe Photoshop and then compared our final output in three metrics- Dice Similarity Coefficient
(DSC), Recognition Rate (RR) and Misidentification rate (MR). Our results showed a great DSC, RR and
an MR to be looked into as future work.

Our algorithm had a DSC more than 0.95%, RR of 90% and MR of 19%.

## References
[1] P2irc-usask: About. https://p2irc.usask.ca/about.php
