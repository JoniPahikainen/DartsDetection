import numpy as np
import cv2
import time 
import math
import pickle


DEBUG = True
num = 6
#cam = cv2.VideoCapture(4)
image_clear = cv2.imread(f"test_set/CLEAR_{num}.jpg")
image_bull = cv2.imread(f"test_set/T20_{num}.jpg")

winName = "test2"

kk = 0


center_dartboard = []
ring_radius = []
transformation_matrix = []

class dartThrow:
    def __init__(self):
        self.base = -1
        self.multiplier = -1
        self.magnitude = -1
        self.angle = -1

class CalibrationData:
    def __init__(self):
        self.top = []
        self.bottom = []
        self.left = []
        self.right = []
        self.init_point_arr = []
        self.center_dartboard = []
        self.ref_angle = []
        self.ring_radius = []
        self.transformationMatrix = []


def drawBoard():
    raw_loc_mat = np.zeros((800, 800, 3))

    cv2.circle(raw_loc_mat, (400, 400), 170 * 2, (255, 255, 255), 1)
    cv2.circle(raw_loc_mat, (400, 400), 160 * 2, (255, 255, 255), 1)
    cv2.circle(raw_loc_mat, (400, 400), 107 * 2, (255, 255, 255), 1)
    cv2.circle(raw_loc_mat, (400, 400), 97 * 2, (255, 255, 255), 1)
    cv2.circle(raw_loc_mat, (400, 400), 16 * 2, (255, 255, 255), 1)
    cv2.circle(raw_loc_mat, (400, 400), 7 * 2, (255, 255, 255), 1)

    sectorangle = 2 * math.pi / 20
    i = 0
    while (i < 20):
        cv2.line(raw_loc_mat, (400, 400), (
            int(400 + 170 * 2 * math.cos((0.5 + i) * sectorangle)),
            int(400 + 170 * 2 * math.sin((0.5 + i) * sectorangle))), (255, 255, 255), 1)
        i = i + 1

    return raw_loc_mat

def dist(x1,y1, x2,y2, x3,y3):
    px = x2-x1
    py = y2-y1

    something = px*px + py*py

    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(something)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    dist = math.sqrt(dx*dx + dy*dy)

    return dist

def DartLocation(x_coord,y_coord):
    try:
            points = []

            calFile = open('calibrationData.pkl', 'rb')
            calData = CalibrationData()
            calData = pickle.load(calFile)
            global transformation_matrix
            transformation_matrix = calData.transformationMatrix
            global ring_radius
            ring_radius.append(calData.ring_radius[0])
            ring_radius.append(calData.ring_radius[1])
            ring_radius.append(calData.ring_radius[2])
            ring_radius.append(calData.ring_radius[3])
            ring_radius.append(calData.ring_radius[4])
            ring_radius.append(calData.ring_radius[5])
            global center_dartboard
            center_dartboard = calData.center_dartboard

            calFile.close()
            dart_loc_temp = np.array([[x_coord, y_coord]], dtype="float32")
            dart_loc_temp = np.array([dart_loc_temp])
            dart_loc = cv2.perspectiveTransform(dart_loc_temp, transformation_matrix)
            new_dart_loc = tuple(dart_loc.reshape(1, -1)[0])

            return new_dart_loc

    except AttributeError as err1:
        print(err1)
        return (-1, -1)

    except NameError as err2:
        print(err2)
        return (-2, -2)
    


#Returns dartThrow (score, multiplier, angle, magnitude) based on x,y location
def DartRegion(dart_loc):
    try:
            height = 800
            width = 800

            global dartInfo

            dartInfo = dartThrow()

            #find the magnitude and angle of the dart
            vx = (dart_loc[0] - center_dartboard[0])
            vy = (center_dartboard[1] - dart_loc[1])

            # reference angle for atan2 conversion
            ref_angle = 81

            dart_magnitude = math.sqrt(math.pow(vx, 2) + math.pow(vy, 2))
            dart_angle = math.fmod(((math.atan2(vy,vx) * 180/math.pi) + 360 - ref_angle), 360)

            dartInfo.magnitude = dart_magnitude
            dartInfo.angle = dart_angle

            angleDiffMul = int((dart_angle) / 18.0)

            print(vx, vy, dart_angle)

            #starting from the 20 points
            if angleDiffMul == 19:
                dartInfo.base = 20
            elif angleDiffMul == 0:
                dartInfo.base = 5
            elif angleDiffMul == 1:
                dartInfo.base = 12
            elif angleDiffMul == 2:
                dartInfo.base = 9
            elif angleDiffMul == 3:
                dartInfo.base = 14
            elif angleDiffMul == 4:
                dartInfo.base = 11
            elif angleDiffMul == 5:
                dartInfo.base = 8
            elif angleDiffMul == 6:
                dartInfo.base = 16
            elif angleDiffMul == 7:
                dartInfo.base = 7
            elif angleDiffMul == 8:
                dartInfo.base = 19
            elif angleDiffMul == 9:
                dartInfo.base = 3
            elif angleDiffMul == 10:
                dartInfo.base = 17
            elif angleDiffMul == 11:
                dartInfo.base = 2
            elif angleDiffMul == 12:
                dartInfo.base = 15
            elif angleDiffMul == 13:
                dartInfo.base = 10
            elif angleDiffMul == 14:
                dartInfo.base = 6
            elif angleDiffMul == 15:
                dartInfo.base = 13
            elif angleDiffMul == 16:
                dartInfo.base = 4
            elif angleDiffMul == 17:
                dartInfo.base = 18
            elif angleDiffMul == 18:
                dartInfo.base = 1
            else:
                dartInfo.base = -300

            for i in range(0, len(ring_radius)):
                if dartInfo.magnitude <= ring_radius[i]:
                    if i == 0:
                        dartInfo.base = 25
                        dartInfo.multiplier = 2
                    elif i == 1:
                        dartInfo.base = 25
                        dartInfo.multiplier = 1
                    elif i == 3:
                        dartInfo.multiplier = 3
                    elif i == 5:
                        dartInfo.multiplier = 2
                    elif i == 2 or i == 4:
                        dartInfo.multiplier = 1
                    break

            if dartInfo.magnitude > ring_radius[5]:
                dartInfo.base = 0
                dartInfo.multiplier = 0

            return dartInfo


    except AttributeError as err1:
        print(err1)
        dartInfo = dartThrow()
        return dartInfo

    except NameError as err2:
        print (err2)
        dartInfo = dartThrow()
        return dartInfo


def getDart():
    global finalScore
    global transformation_matrix

    debug_img = drawBoard()
    kko = 0
    finalScore = 0
    count = 0
    breaker = 0
    success = 1
    x = 1500
    #cv2.imshow("Test", image_clear)
    img_copy = image_bull.copy()
    t = cv2.cvtColor(image_clear, cv2.COLOR_RGB2GRAY)

    while success:
        time.sleep(2)
        #success,image = cam.read()
        kko += 1
        t_plus = cv2.cvtColor(image_bull, cv2.COLOR_RGB2GRAY)
        cv2.imwrite("images/1t.jpg", t)
        cv2.imwrite("images/1t_plus.jpg", t_plus)
        dimg = cv2.absdiff(t, t_plus)
        #cv2.imshow(f'Difference Image', dimg)
        cv2.imwrite("images/1.jpg", dimg)
        cv2.waitKey(1)

        blur = cv2.GaussianBlur(dimg,(5,5),0)
        blur = cv2.bilateralFilter(blur,9,75,75)
        ret, thresh = cv2.threshold(blur, 30, 255, 0)
        #cv2.imshow(f'Thresh Image', thresh)
        cv2.imwrite("images/thresh.jpg", thresh)
        image = image_bull.copy()
        



        cv2.waitKey(1)
        print(f"success: {success} {kko} {cv2.countNonZero(thresh)}")
        if cv2.countNonZero(thresh) > x and cv2.countNonZero(thresh) < 15000:
            time.sleep(0.2)
            t_plus = cv2.cvtColor(image_bull, cv2.COLOR_RGB2GRAY)

            cv2.imwrite("images/tplus_eka.jpg", t_plus)
            dimg = cv2.absdiff(t, t_plus)
            cv2.imwrite("images/dimg2.jpg", dimg)
            

            kernel = np.ones((8, 8), np.float32) / 40
            blur = cv2.filter2D(dimg, -1, kernel)
            ret, thresh = cv2.threshold(blur, 30, 255, 0)
            cv2.imwrite("images/blurrrr.jpg", blur)
            edges = cv2.Canny(thresh, 50, 150)
            cv2.imwrite("images/edges.jpg", edges)

            edges = cv2.goodFeaturesToTrack(blur,640,0.0008,3,mask=None, blockSize=3, useHarrisDetector=1, k=0.06) # k=0.08
            edges2 = cv2.Canny(blur, 100,200)
            #print(f"edges: {edges}")
            #print(f"edges2: {edges2}")
            #cv2.imwrite("images/edges.jpg", edges)
            #cv2.imwrite("images/edges2.jpg", edges2)
            corners = np.int64(edges)
            testimg = blur.copy()
            t_plus_copy = t_plus.copy()

            cornerdata = []
            tt = 0
            mean_corners = np.mean(corners, axis=0)
            for i in corners:
                xl, yl = i.ravel()
                if abs(mean_corners[0][0] - xl) > 180:
                    cornerdata.append(tt)
                if abs(mean_corners[0][1] - yl) > 120:
                    cornerdata.append(tt)
                tt += 1

            print(f"corners: {corners.size}")
            corners_new = np.delete(corners, [cornerdata], axis=0)
            print(f"corners_new: {corners_new.size}")


            rows,cols = dimg.shape[:2]
            [vx,vy,x,y] = cv2.fitLine(corners_new,cv2.DIST_HUBER, 0,0.1,0.1)
            lefty = int(((-x[0] * vy[0] / vx[0]) + y[0]))
            righty = int((((cols - x[0]) * vy[0] / vx[0]) + y[0]))

            cv2.line(image, (0, lefty), (cols - 1, righty), (0, 255, 0), 2)
            cv2.imwrite("images/line.jpg", image)
            cv2.line(thresh, (0, lefty), (cols - 1, righty), (0, 255, 0), 2)
            cv2.imwrite("images/thresh.jpg", thresh)

            cornerdata = []
            tt = 0
            km = 0
            for i in corners_new:
                #print(f"i: {i}")
                xl,yl = i.ravel()
                cv2.circle(testimg,(xl,yl),3,255,-1)
                #print(f"xl: {xl},yl: {yl}")
                distance = dist(0,lefty, cols-1,righty, xl,yl)
                if distance < 40:
                    print(f"Distance: {distance}")
                    #cv2.circle(testimg,(xl,yl),3,255,-1)
                    km += 1
                else:
                    cornerdata.append(tt)
                tt += 1

            print(f"km: {km}, tt: {tt}")
            corners_final = np.delete(corners_new, [cornerdata], axis=0)

            ret, thresh = cv2.threshold(blur, 60, 255, 0)

            if cv2.countNonZero(thresh) > 15000:
                continue

            x,y,w,h = cv2.boundingRect(corners_final)

            cv2.rectangle(img_copy,(x,y),(x+w,y+h),(0,255,0),1)

            breaker += 1
            maxloc = np.argmax(corners_final, axis=0)
            locationofdart = corners_final[maxloc]

            try:
                cornerdata = []
                tt = 0
                for i in corners_final:
                    xl, yl = i.ravel()
                    distance = abs(locationofdart.item(0) - xl) + abs(locationofdart.item(1) - yl)
                    if distance < 40:
                        tt += 1
                    else:
                        cornerdata.append(tt)

                if tt < 3:
                    corners_temp = cornerdata
                    maxloc = np.argmax(corners_temp, axis=0)
                    locationofdart = corners_temp[maxloc]
                    print("### used different location due to noise!")

                print("locationofdart after processing:", locationofdart)

                cv2.circle(img_copy, (locationofdart.item(0),locationofdart.item(1)), 10,(0, 0, 0),2, 8)
                cv2.circle(img_copy, (locationofdart.item(0), locationofdart.item(1)), 2, (0, 0, 0), 2, 8)

                dartloc = DartLocation(locationofdart.item(0), locationofdart.item(1))
                dartInfo = DartRegion(dartloc)

            except Exception as e:
                print("Something went wrong in finding the darts location!")
                print("Exception:", e)
                continue

            print (dartInfo.base, dartInfo.multiplier)

            if breaker == 1:
                cv2.imwrite("images/frame2.jpg", testimg)
            elif breaker == 2:
                cv2.imwrite("images/frame3.jpg", testimg)
            elif breaker == 3:
                cv2.imwrite("images/frame4.jpg", testimg)

            t = t_plus
            finalScore += (dartInfo.base * dartInfo.multiplier)

            if DEBUG:
                loc_x = dartloc[0] #400 + dartInfo.magnitude * math.tan(dartInfo.angle * math.pi/180)
                loc_y = dartloc[1] #400 + dartInfo.magnitude * math.tan(dartInfo.angle * math.pi/180)
                cv2.circle(debug_img, (int(loc_x), int(loc_y)), 2, (0, 255, 0), 2, 8)
                cv2.circle(debug_img, (int(loc_x), int(loc_y)), 6, (0, 255, 0), 1, 8)
                string = "" + str(dartInfo.base) + "x" + str(dartInfo.multiplier)
                cv2.rectangle(debug_img, (600, 700), (800, 800), (0, 0, 0), -1)
                cv2.putText(debug_img, string, (600, 750), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, 8)
                # HERE2 cv2.namedWindow("cv2.WINDOW_NORMAL", cv2.WINDOW_NORMAL)
                # HERE2 cv2.namedWindow("raw", cv2.WINDOW_NORMAL)
                # HERE2 cv2.namedWindow("test", cv2.WINDOW_NORMAL)
                # HERE2 cv2.imshow("debug_img", debug_img)
                # HERE2 cv2.imshow("raw", t_plus_copy)
                # HERE2 cv2.imshow("test", testimg)
                cv2.imwrite("images/debug_img.jpg", debug_img)
                cv2.imwrite("images/t_plus_copy.jpg", img_copy)
                cv2.imwrite("images/testimg.jpg", testimg)
            else:
                # HERE2 cv2.imshow("testimg", testimg)
                cv2.imwrite("images/else_testimg.jpg", testimg)

            success = False

        elif cv2.countNonZero(thresh) < 35000:
            continue

        elif cv2.countNonZero(thresh) > 35000:
            break

        key = cv2.waitKey(10)
        if key == 27:
            cv2.destroyWindow(winName)
            break

        count += 1


dartInfo = dartThrow()


if __name__ == '__main__':
    print ("Welcome to darts!")
    getDart()
    #getTransformation()