import numpy as np
import cv2


class HeadTracker:
        
    def __init__(self):
        print("Tracker funcionando!")
        # initialize persistent drawing mask and random colors
        self.mask = None
        self.randomColors = None
        
    def lucasKanade(self, frameGrayPrev, frameGrayCurrent, frame):
        # Setup Parameters
        shiTomasiCornerParams = dict(maxCorners=20,
                                 qualityLevel=0.3,
                                 minDistance=50,
                                 blockSize=7)
        lucasKanadeParams = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


        cornersPrev = cv2.goodFeaturesToTrack(frameGrayPrev, mask=None, **shiTomasiCornerParams)
        # find features to track and compute optical flow
        if cornersPrev is None:
            return frame

        cornersCur, foundStatus, _ = cv2.calcOpticalFlowPyrLK(frameGrayPrev, frameGrayCurrent, cornersPrev, None, **lucasKanadeParams)

        if cornersCur is not None and foundStatus is not None:
            good = foundStatus.ravel() == 1
            cornersMatchedCur = cornersCur[good]
            cornersMatchedPrev = cornersPrev[good]
        else:
            return frame

        # initialize mask and random colors if needed
        if self.mask is None:
            self.mask = np.zeros_like(frame)
        if self.randomColors is None:
            max_c = shiTomasiCornerParams.get("maxCorners", 20)
            self.randomColors = np.random.randint(0, 255, (max_c, 3), dtype=np.uint8)
        
        for i, (curCorner, prevCorner) in enumerate(zip(cornersMatchedCur, cornersMatchedPrev)):
            xCur, yCur = curCorner.ravel()
            xPrev, yPrev = prevCorner.ravel()
            self.mask = cv2.line(self.mask, (int(xCur), int(yCur)), (int(xPrev), int(yPrev)), self.randomColors[i].tolist(), 2)
            frame = cv2.circle(frame, (int(xCur), int(yCur)), 5, self.randomColors[i].tolist(), -1)
            img = cv2.add(frame, self.mask)

        cornersPrev=cornersMatchedCur.reshape(-1,1,2)
        
        return img