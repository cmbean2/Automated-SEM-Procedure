These scripts perform the automated acquistion of SEM Images using the FEI Autoscript Functionality 
to capture a high resolution grid of images based on user specified settings.

There are two scripts:
- InitialStep
- AlignedSteps

Both scripts are designed to have the user intialize the microscope and find the desired starting location. 
Then upon starting the script will complete the acquistion of the images in a snake like pattern.
It will perform autofocusing and autoastigmatism before every image, and then at the beginning and every 10 images 
it will also adjust the brightness and contract as well.

Similarly, the script will display a message on the screen indicating that a automated session 
is in progress, show an estimated end time, and the name and contact of the individual running the session. 

The AlignedSteps script is different from the InitialStep script as it also requires a folder of already taken 
images to be provided as it uses template matching to align the images to the previous set of images. 
It is recommended to use the most immediate previous set of images for the alignment and do additional 
alignment of images separately to get the best results. 
