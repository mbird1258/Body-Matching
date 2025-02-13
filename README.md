# Body World Eye Mapping
Blog post: [https://matthew-bird.com/blogs/Body-World-Eye-Mapping.html](https://matthew-bird.com/blogs/Body-World-Eye-Mapping.html)

### Youtube videos
[![Demo Video 1](https://img.youtube.com/vi/EAv7olLFaek/0.jpg)](https://www.youtube.com/watch?v=EAv7olLFaek)
[![Demo Video 2](https://img.youtube.com/vi/oWKc6HfKZ_M/0.jpg)](https://www.youtube.com/watch?v=oWKc6HfKZ_M)

## Premise
My plan for this project was to make use of 2 cameras and a variety of methods to detect/track objects in videos in order to map them in 3D. This way, it would be possible to make a 3D video that could be viewed frame by frame or as in the form of a continuous video in order to analyse volleyball games. This was largely motivated by my interest in and good body type for volleyball but terrible skills to accompany. 

## Recording the Video
The stand used was pretty flimsy and cracked in a few areas, which ended up causing me quite a bit of trouble. For aligning the frames of the video, I suggest taking a fast movement such as an eye blinking to align the two videos. In addition to this, it can be useful to run the following command to show the exact time and frame of a video (at least for quicktime player on mac): 
ffmpeg -i input.mp4 -c copy -timecode 00:00:00:00 output.mp4

It’s important to make sure that both videos have the same framerate as well. The frame rate can be changed with the following command:
ffmpeg -i input.mp4 -c copy -r [fps] output.mp4

## Body
The body detection was done through the use of [RTMO](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo), which I found to be much more robust than other libraries like mediapipe. It was a bit of a pain to set up until I found [RTMLib](https://github.com/Tau-J/rtmlib)(linked on the RTMO github which I annoyingly missed), but using RTMLib was simple enough. 

<ins>Image of RTMO applied to image of game</ins>

![image](https://github.com/user-attachments/assets/4e34e29f-be31-40d1-b692-df5080895cd4)

To accomplish body matching, I took each body and normalized the average of the positions of each joint to 0. Then, I calculated the smallest distance between corresponding joints between each body in the first image and each body in the second image and took the match with the lowest error. This works largely because the distance between the two cameras was largely insubstantial compared to the distance the bodies were from the cameras, meaning that the rotational and scaling effect of a different camera position was largely minimized. 

<ins>Matched bodies</ins>

<img width="200" alt="" src="https://github.com/user-attachments/assets/824dab29-e5db-445a-bd41-68f499dc222d">
<img width="200" alt="" src="https://github.com/user-attachments/assets/2f5568c4-b37a-4b69-bbeb-c6df743bdf17">
<img width="200" alt="" src="https://github.com/user-attachments/assets/cd0d73a1-4fa2-465a-ab7f-85ee55816c69">
<img width="200" alt="" src="https://github.com/user-attachments/assets/dcb612f9-fc88-4346-be03-7ce23d16e42f">
<img width="200" alt="" src="https://github.com/user-attachments/assets/695fa342-86ac-4197-be19-3802836201b3">
<img width="200" alt="" src="https://github.com/user-attachments/assets/b3baacf5-a2f9-45f4-994b-74f806784684">
<img width="200" alt="" src="https://github.com/user-attachments/assets/72e1ce2c-aadc-400c-9aa8-1264ec850c68">
<img width="200" alt="" src="https://github.com/user-attachments/assets/0942ccb1-addd-4b26-9891-b5eaa79dab93">

## Volleyball
In order to track the volleyball, I went through a lot of solutions online, including a large amount of machine learning models on roboflow and some algorithms online on GitHub, but they were all either not robust enough or I encountered a lot of issues implementing them. 

In the end, I decided to come up with my own solution, which involved creating a mask of all moving objects in the frame, creating a mask of all objects fitting within the correct color range, and taking the largest contour/blob of pixels that satisfy both conditions. More specifically, to detect motion within the image, I took the median of the last 60 or so frames, giving us an image of the court without any people or balls, and took the difference in colour between this median and the current frame. 

<ins>Input</ins>

<img width="300" alt="" src="https://github.com/user-attachments/assets/7e0de23f-ec21-415d-99fb-8aeeec2cbe24">

<ins>Median</ins>

<img width="300" alt="" src="https://github.com/user-attachments/assets/a69cbb11-b2bc-475b-ac3b-73f1d2eea2bc">

<ins>Movement mask</ins>

<img width="300" alt="" src="https://github.com/user-attachments/assets/46401ff6-493c-44b8-ac2b-f0ea40fb09fd">

<ins>Colour Mask</ins>

<img width="300" alt="" src="https://github.com/user-attachments/assets/88d44944-50c1-4162-ac49-b72e99e7fb03">

<ins>Combined Mask</ins>

<img width="300" alt="" src="https://github.com/user-attachments/assets/9aabf288-b231-4349-9515-1180281ad5df">

<ins>Ball Detection</ins>

<img width="300" alt="" src="https://github.com/user-attachments/assets/b7b13dec-2986-4a71-b652-54dceed39a40">
<img width="300" alt="" src="https://github.com/user-attachments/assets/64254fe9-d483-4618-b264-8cc07890bd32">

## Eyes
### Finding the center of the eye
I had originally intended to use the eye matching for each of the individual players on the court, but considering that almost all the players’ eyes are hidden by their heads and the resolution and robustness of the ML algorithms I used would lead to results that are too inconsistent to rely upon. However, I still tried to implement it as a sort of separate mini-project to track the position of the eye’s focus in short range. 

The math behind finding the centers of the spheres isn’t too difficult. The formula for a sphere, given that the center of the sphere is $(x,y,z)$, is:

By expanding, we get the following equation:

$(x_1-x)^2+(y_1-y)^2+(z_1-z)^2=r^2$

Since we are solving for $x$, $y$, and $z$, we can plug in values from a point we know lies on the surface of the sphere (thus satisfying the equation) in for $x_1$, $y_1$, and $z_1$, thus allowing us to treat $x_1$, $y_1$, and $z_1$ as constants. Solving for $x$, $y$, and $z$ on the left, we get:

$x_1^2-2xx_1+x^2+y_1^2-2yy_1+y^2+z_1^2-2zz_1+z^2=r^2$

Crucially, since we want to remove the $x^2$, $y^2$, $z^2$, and $r^2$ from the equation, if we now consider the 2 equations defining 2 points and subtract them from each other, we get:

$2xx_1+2yy_1+2zz_1-2xx_2-2yy_2-2zz_2=x_1^2+y_1^2+z_1^2-x_2^2-y_2^2-z_2^2$

Since we have 3 unknowns, we need 3 equations, thus requiring 4 points. 

$2xx_1+2yy_1+2zz_1-2xx_2-2yy_2-2zz_2=x_1^2+y_1^2+z_1^2-x_2^2-y_2^2-z_2^2$
$2xx_1+2yy_1+2zz_1-2xx_3-2yy_3-2zz_3=x_1^2+y_1^2+z_1^2-x_3^2-y_3^2-z_3^2$
$2xx_1+2yy_1+2zz_1-2xx_4-2yy_4-2zz_4=x_1^2+y_1^2+z_1^2-x_4^2-y_4^2-z_4^2$

Writing this in matrix form such that we can solve it, we get the following:

<img width="592" alt="image" src="https://github.com/user-attachments/assets/a4e53df7-1b7b-48bc-9bc8-84f9e3e09028" />

Solving this matrix then returns the coordinates of the center of the sphere. Doing this for every combination of 4 points out of the points we have on the surface of the eye and applying RANSAC with the radius being constrained to around 1.15 cm gives us a pretty good approximation of the center of the eye. 

The points of the surface of the eye can be determined with two cameras, by first using machine learning to find corresponding points on the eye on each image, then finding the intersection of the corresponding two lines drawn in 3D space.

### Iris triangulation
In order to find the center of the iris, we once again employ machine learning to detect the position of the iris on screen in images from two cameras and follow the same procedure in the above paragraph to triangulate the position of the iris. 

### Focus triangulation
In order to find the point in 3D space that the person is focusing on, we can draw a line from the center of each eye to the corresponding iris position in 3D space, then find the intersection between these two lines. 

### Results
It ended up not working :( . None of the off the shelf face landmarking machine learning models were capable of finding points on my eyes that were accurate enough across both images to triangulate in 3d and use to get a clean sphere. Every time, I would end up with a sphere roughly two times too big for a human eye, and oftentimes the eye landmarks wouldn’t even be on my eye in the 2d images. Overall, I tried multiple models including mediapipe, dlib, openface, retinaface, MTCNN, SPIGA, yolov8-face, RTMPose wholebody, and more, and none seemed to work well enough. The fact that these models aren’t perfectly accurate and my method to triangulate focus from images has a very low tolerance for inaccuracy led to results that were unfortunately completely unusable. I also tried to use a [different sphere finding algorithm](https://jekel.me/2015/Least-Squares-Sphere-Fit/), and it worked quite well but still couldn’t overcome the inaccurate eye landmarks and couldn’t constrain the eye radius like RANSAC could. 

## Mapping Court Points
The court points’ position on each image was determined through manual input, which I decided would be best given that it only needs to be done once for a long video. In order to get the full court when most of the court was cropped out by the restricted cameras’ field of views, I estimated each point defining the court from just 3. This is done by manually inputting 3 of the court points and finding their position in 3D, rotating the 3 points to have the same z value in space, rotating and translating hard coded 3d points that define the entire volleyball court such that the 3 inputted points and the corresponding hard coded points align, and then reversing the rotation we performed to make the 3 points have the same z value in space. 

## Floundering Around With Mapping
Originally, mapping shouldn’t have been much of an issue. Given two images with a field of views of both and the offset in position between the two, using the formula found [here(shortest line between two lines in 3D)](https://paulbourke.net/geometry/pointlineplane/). Thus, since we have the matched points of each body’s joints and the volleyball, we could just find their most likely position in 3D space. However, this didn’t really go as planned. 

The first hurdle was finding the field of view of the ipad and iphone that I used. I ended up sinking hours into different methods before finding [this](https://www.panohelp.com/lensfov.html), which I think gave the best results. I also spent an hour or two checking the camera offset of the cameras in my camera setup over and over again, changing absolutely nothing. 

The next hurdle was the shoddy 3D printed setup I used for positioning the cameras stationary relative to each other, which ended up leaving one camera’s orientation different from the other when the code assumed they were identical. This tiny offset in rotation led to a large offset in position of pixels in the image, which was further exacerbated by the large distance between the camera and the bodies and the low distance between cameras. 

In order to solve this, I used something called a homography matrix. The homography matrix effectively transforms, rotates, shears, and warps the image such that given a rectangle can be remapped to any quadrilateral. This allows the changing of the perspective of an image, thus why it is sometimes called the perspective projection matrix, which if calibrated correctly could solve any problems to do with rotation or incorrect positioning of the second camera. To solve for the homography matrix which has 8 degrees of freedom, we need 4 points on the image and the 4 new points we want to obtain. This is obtained by finding the position of [court points in 3D](https://www.geogebra.org/3d/k7waft2t) and projecting those onto the image plane of the second camera as the target points. Then, using those target points and the originally found points of the court on the image, we can [calculate](https://math.stackexchange.com/a/2619023) a homography matrix. Unfortunately, this only improved the results and failed to magically solve all my problems. 

My only possible explanation remaining for the triangulation still not working is that the terrible recording setup, far distance of the court from the camera, relatively small distance between the two cameras, and natural inaccuracy all work in conjunction to make the project fail. 

## Conclusion
The original intent of the project - mapping an entire volleyball game in 3D - failed. Regardless, [the project is up on GitHub](https://github.com/mbird1258/Body-World-Eye-Mapping), as others can use it as a reference, and if hardware was really the issue, the software is probably somewhat correct. In addition to this, the project successfully accomplished tracking of volleyballs and matching bodies between two images, which each have been posted as individual GitHub projects too. [Volleyball](https://github.com/mbird1258/Volleyball-Tracking) [Body matching](https://github.com/mbird1258/Body-Matching) [Eye focus mapping](https://github.com/mbird1258/Eye-Focus-Mapping)

In the future I might come back to this project with a proper recording setup since it has quite a bit of potential in volleyball analytics, as I had originally planned to make a nice frontend for it and have it calculate the ball velocity and acceleration, body joint velocity and acceleration, jump height, spike speed, compare bumping/setting/spiking form etc. 
