import utils
import numpy as np
import matplotlib.pyplot as plt
import cv2

class BodyManager:
    def BodyMatching(self, img1, img2, debug=False):
        # [Bodies] [[Joints]] [[[x, y]]] [# bodies, 17, 2]
        img1arr = utils.BodyPose(img1)
        img2arr = utils.BodyPose(img2)
        if img1arr.size == 0 or img2arr.size == 0:
            return np.array([])

        norm = lambda x: (x - np.mean(x, axis=1)[:, np.newaxis, :])#/(np.max(x, axis=1)[:, np.newaxis, :]-np.min(x, axis=1)[:, np.newaxis, :])
        OffsetMatrix = np.mean((norm(img1arr)[:, np.newaxis, :, :] - norm(img2arr)[np.newaxis, :, :, :])**2, axis=(2,3))
        
        # Normalize to center around (0,0)
        img1arr -= np.array(img1.shape)[[1, 0]][np.newaxis, np.newaxis, :]/2
        img2arr -= np.array(img2.shape)[[1, 0]][np.newaxis, np.newaxis, :]/2
        
        out = []
        while True:
            ind1, ind2 = np.unravel_index(np.argmin(OffsetMatrix, axis=None), OffsetMatrix.shape)
            
            # Exit if offset too high
            if OffsetMatrix[ind1, ind2] == np.inf:
                break

            if debug:
                connections = utils.BodyPoseConnections()
                plt.imshow(img1, alpha=0.5)
                for ind1_, ind2_ in connections:
                    x1, y1 = img1arr[ind1][ind1_] + np.array([*img1.shape[:-1], 0])[[1, 0]]/2
                    x2, y2 = img1arr[ind1][ind2_] + np.array([*img1.shape[:-1], 0])[[1, 0]]/2

                    plt.plot([x1, x2], [y1, y2])
                plt.show()
                plt.imshow(img2, alpha=0.5)
                for ind1_, ind2_ in connections:
                    x1, y1 = img2arr[ind2][ind1_] + np.array([*img2.shape[:-1], 0])[[1, 0]]/2
                    x2, y2 = img2arr[ind2][ind2_] + np.array([*img2.shape[:-1], 0])[[1, 0]]/2

                    plt.plot([x1, x2], [y1, y2])
                plt.show()
            
            mask = np.ones_like(OffsetMatrix).astype(np.bool_)
            mask[ind1] = False
            mask[:, ind2] = False
            OffsetMatrix[~mask] = np.inf

            out.append([img1arr[ind1], img2arr[ind2]])
        
        return out

    def DrawBodies(self, img1, img2, bodies):
        img1 = (np.copy(img1)/2).astype(np.uint8)
        img2 = (np.copy(img2)/2).astype(np.uint8)

        colors = utils.COLORS
        connections = utils.BodyPoseConnections()

        h1, w1, _ = img1.shape
        h2, w2, _ = img2.shape
        
        for BodyInd, (Body1, Body2) in enumerate(bodies):
            color = colors[BodyInd%len(colors)]
            
            for ind1, ind2 in connections:
                x1, y1 = (Body1[ind1] + np.array([w1, h1])/2).astype(int)
                x2, y2 = (Body1[ind2] + np.array([w1, h1])/2).astype(int)

                cv2.line(img1, [x1, y1], [x2, y2], color=color, thickness=5)

            for ind1, ind2 in connections:
                x1, y1 = (Body2[ind1] + np.array([w2, h2])/2).astype(int)
                x2, y2 = (Body2[ind2] + np.array([w2, h2])/2).astype(int)

                cv2.line(img2, [x1, y1], [x2, y2], color=color, thickness=5)
        
        out = np.zeros((max(h1, h2), w1+w2,3), np.uint8)

        out[:h1, :w1] = img1/2
        out[:h2, w1:w1+w2] = img2/2

        return out

    def ProcessVideo(self, vid1, StartFrame1, vid2, StartFrame2, FileName="BodiesOut1.mp4", length=False):
        cap1 = cv2.VideoCapture(vid1)
        cap1.set(cv2.CAP_PROP_POS_FRAMES, StartFrame1-1)
        cap2 = cv2.VideoCapture(vid2)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, StartFrame2-1)

        res1, frame1 = cap1.read()
        res2, frame2 = cap2.read()
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        if not res1 or not res2:
            return

        video=cv2.VideoWriter(FileName,
                              cv2.VideoWriter_fourcc(*'mp4v'), 
                              cap1.get(cv2.CAP_PROP_FPS),
                              [frame1.shape[1]+frame2.shape[1], max(frame1.shape[0], frame2.shape[0])])
        
        t = 0
        while True:
            print(f"frame: {t}", end="\r")

            if length:
                if t >= length:
                    break
            t += 1
            
            bodies = self.BodyMatching(frame1, frame2)
            img = self.DrawBodies(frame1, frame2, bodies)
            video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            res1, frame1 = cap1.read()
            res2, frame2 = cap2.read()
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            if not res1 or not res2:
                break
        
        video.release()