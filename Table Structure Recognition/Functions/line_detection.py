import cv2
import numpy as np

# Input : Image
# Output : hor,ver 
def line_detection(image):

    img_name = "mt04"
    #res_path = "/home/elimen/Data/cascade-tabnet_pytorch/Data Preparation/test_results/"
    res_path = "/home/junlin/Git/github/cascade-tabnet_pytorch/Data Preparation/test_results/"


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 1)
    cv2.imwrite(res_path + img_name + '_adathres.jpg',bw)
    bw = cv2.bitwise_not(bw)
    cv2.imwrite(res_path + img_name + '_bitwise_not.jpg',bw)

    '''
    tmp operation:
    Read txt lines for bbox location.
    '''
    zeros = np.zeros((image.shape), dtype=np.uint8)
    #bbox_file = open('/home/elimen/Data/cascade-tabnet_pytorch/Table Structure Recognition/Functions/mt01_result_bbox.txt', 'r')
    bbox_file = open('/home/junlin/Git/github/cascade-tabnet_pytorch/Table Structure Recognition/Functions/mt04_result_bbox.txt', 'r')
    bbox_lines = bbox_file.readlines()
    bboxes_loc = []
    for line in bbox_lines:
        line = line.replace('[','')
        line = line.replace(']','')
        line = line.replace(' ','')
        x1 = int(line.split('.')[0])
        y1 = int(line.split('.')[1])
        x2 = int(line.split('.')[2])
        y2 = int(line.split('.')[3])
        bbox = [x1,y1,x2,y2]
        bboxes_loc.append(bbox)
        offset_x =  int((bbox[3] - bbox[1])*0.12)
        offset_y =  int((bbox[3] - bbox[1])*0.15)
        zeros_mask = cv2.rectangle(bw, (bbox[0]-offset_x, bbox[1]-offset_y), (bbox[2]+offset_x, bbox[3]+offset_y), color=(0,0,0), thickness=-1)

    cv2.imwrite(res_path + img_name + '_zeros_mask.jpg',zeros_mask)

    ###########################################
    horizontal = bw.copy()
    vertical = bw.copy()
    img = image.copy()
    # [horizontal lines]
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))

    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    cv2.imwrite(res_path + img_name + '_erodeH.jpg',horizontal)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    cv2.imwrite(res_path + img_name + '_dilateH.jpg',horizontal)
    horizontal = cv2.dilate(horizontal, (1,1), iterations=5)
    horizontal = cv2.erode(horizontal, (1,1), iterations=5)

    ## Uncomment to visualize highlighted Horizontal lines
    cv2.imwrite(res_path + img_name + '_horizontal.jpg',horizontal)
    # cv2.imshow("horizontal",horizontal)
    # cv2.waitKey(0)

    # HoughlinesP function to detect horizontal lines
    hor_lines = cv2.HoughLinesP(horizontal,rho=1,theta=np.pi/180,threshold=100,minLineLength=30,maxLineGap=3)
    if hor_lines is None:
        return None,None
    temp_line = []
    for line in hor_lines:
        for x1,y1,x2,y2 in line:
            temp_line.append([x1,y1-5,x2,y2-5])

    # Sorting the list of detected lines by Y1
    hor_lines = sorted(temp_line,key=lambda x: x[1])

    ## Uncomment this part to visualize the lines detected on the image ##
    # print(len(hor_lines))
    for x1, y1, x2, y2 in hor_lines:
        cv2.line(image, (x1,y1), (x2,y2), (0, 255, 0), 1)

    cv2.imwrite(res_path + img_name + '_linesH.jpg',image)
    # print(image.shape)
    # cv2.imshow("image",image)
    # cv2.waitKey(0)
    ####################################################################

    ## Selection of best lines from all the horizontal lines detected ##
    lasty1 = -111111
    lines_x1 = []
    lines_x2 = []
    hor = []
    i=0
    for x1,y1,x2,y2 in hor_lines:
        if y1 >= lasty1 and y1 <= lasty1 + 10:
            lines_x1.append(x1)
            lines_x2.append(x2)
        else:
            if (i != 0 and len(lines_x1) is not 0):
                hor.append([min(lines_x1),lasty1,max(lines_x2),lasty1])
            lasty1 = y1
            lines_x1 = []
            lines_x2 = []
            lines_x1.append(x1)
            lines_x2.append(x2)
            i+=1
    hor.append([min(lines_x1),lasty1,max(lines_x2),lasty1])

    
    imgtmp = img.copy()
    for x1, y1, x2, y2 in hor:
        cv2.line(imgtmp, (x1,y1), (x2,y2), (0, 255, 0), 1)
    cv2.imwrite(res_path + img_name + '_linesH_2.jpg',imgtmp)
    #####################################################################


    # [vertical lines]
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))

    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    vertical = cv2.dilate(vertical, (1,1), iterations=8)
    vertical = cv2.erode(vertical, (1,1), iterations=7)

    ######## Preprocessing Vertical Lines ###############
    cv2.imwrite(res_path + img_name + '_vertical.jpg',vertical)
    # cv2.imshow("vertical",vertical)
    # cv2.waitKey(0)
    #####################################################

    # HoughlinesP function to detect vertical lines
    # ver_lines = cv2.HoughLinesP(vertical,rho=1,theta=np.pi/180,threshold=20,minLineLength=20,maxLineGap=2)
    ver_lines = cv2.HoughLinesP(vertical, 1, np.pi/180, 20, np.array([]), 20, 2)
    if ver_lines is None:
        return None,None
    temp_line = []
    for line in ver_lines:
        for x1,y1,x2,y2 in line:
            temp_line.append([x1,y1,x2,y2])

    # Sorting the list of detected lines by X1
    ver_lines = sorted(temp_line,key=lambda x: x[0])

    ## Uncomment this part to visualize the lines detected on the image ##
    # print(len(ver_lines))
    for x1, y1, x2, y2 in ver_lines:
        cv2.line(image, (x1,y1-5), (x2,y2-5), (0, 255, 0), 1)

    cv2.imwrite(res_path + img_name + '_lines.jpg',image)
    # print(image.shape)
    # cv2.imshow("image",image)
    # cv2.waitKey(0)
    ####################################################################

    ## Selection of best lines from all the vertical lines detected ##
    lastx1 = -111111
    lines_y1 = []
    lines_y2 = []
    ver = []
    count = 0
    lasty1 = -11111
    lasty2 = -11111
    for x1,y1,x2,y2 in ver_lines:
        if x1 >= lastx1 and x1 <= lastx1 + 15 and not (((min(y1,y2)<min(lasty1,lasty2)-20 or min(y1,y2)<min(lasty1,lasty2)+20)) and ((max(y1,y2)<max(lasty1,lasty2)-20 or max(y1,y2)<max(lasty1,lasty2)+20))):
            lines_y1.append(y1)
            lines_y2.append(y2)
            # lasty1 = y1
            # lasty2 = y2
        else:
            if (count != 0 and len(lines_y1) is not 0):
                ver.append([lastx1,min(lines_y2)-5,lastx1,max(lines_y1)-5])
            lastx1 = x1
            lines_y1 = []
            lines_y2 = []
            lines_y1.append(y1)
            lines_y2.append(y2)
            count += 1
            lasty1 = -11111
            lasty2 = -11111
    ver.append([lastx1,min(lines_y2)-5,lastx1,max(lines_y1)-5])
    #################################################################

    zerostmp = np.zeros((image.shape), dtype=np.uint8)
    ############ Visualization of Lines After Post Processing ############
    for x1, y1, x2, y2 in ver:
        cv2.line(img, (x1,y1), (x2,y2), (0, 0,255), 1)
        cv2.line(zerostmp, (x1,y1), (x2,y2), (0, 0,255), 1)

    for x1, y1, x2, y2 in hor:
        cv2.line(img, (x1,y1), (x2,y2), (0, 0, 255), 1)
        cv2.line(zerostmp, (x1,y1), (x2,y2), (0, 0, 255), 1)
    
    cv2.imwrite(res_path + img_name + '_interresults.jpg',zerostmp)
    
    #cv2.imshow("image",img)
    #cv2.waitKey(0)
    #######################################################################

    return hor,ver,img

hori, vert, dst = line_detection(cv2.imread('/home/junlin/Git/github/cascade-tabnet_pytorch/Data Preparation/test_images/mt04.png'))
cv2.imwrite('/home/junlin/Git/github/cascade-tabnet_pytorch/Data Preparation/test_results/mt04_results.jpg',dst)