import cv2
import numpy as np
import pickle





class Video:

    def __init__(self):
        self.frame_size = (600,900) # height, width
        self.image = None # image to be detected
        self.rotation = [0] # angle variation of image shown in the frames
        self.size = [50] # size variation of image shown in the frames
        self.margin = None # margin will be decided once size variation above is decided
        self.moves = ['up','down','left','right'] # directions toward which image can move in each step
        self.trajectory = [1,1,1,1] # accumulated moves which will be reset once the image touches the edeg of the frame       
        self.frames = [] # each frame in the list is a image on a noisy background
        self.boxes = [] # each element in the list consist of 2 coordinates that form a box to capture the image

    def set_frame_size(self, height, width):
        self.frame_size = (height, width)

    def set_image(self, path):
        # converts the image to grey scale and cut the edge from it
        image = cv2.imread(path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image = cv2.Canny(image,100,200)
        kernel = np.ones((5,5),np.uint8)
        self.image = cv2.dilate(image,kernel,iterations = 1)

    def set_rotation(self,angle_list):
        assert sum([-360<ang<360 and isinstance(ang, int) for ang in angle_list]) == len(angle_list)
        self.rotation = sorted(angle_list)

    def set_size(self, var_list):
        max_size = min(self.frame_size)//2
        assert sum([0<var<max_size and isinstance(var, int) for var in var_list]) == len(var_list)
        self.size = sorted(var_list)
        self.margin = self.size[-1] - self.size[0] + 10

    def generate_frames(self, frames, start_pos, step):

        pos = start_pos

        for i in range(frames):            
            bg     = self._background()            
            size   = np.random.choice(self.size, 1)[0]
            angle  = np.random.choice(self.rotation, 1)[0]    
            target = self._sample(size, angle)               
            pos    = (pos[0], pos[1], pos[0]+size, pos[1]+size)    
            pos    = self._next_position(pos, step)              
            result = self._combine(bg, target, pos)
            
            self.frames.append(result)
            self.boxes.append((pos[0]/self.frame_size[1], pos[1]/self.frame_size[0], 
                               pos[2]/self.frame_size[1], pos[3]/self.frame_size[0]))

    def output_frames(self, path, to_pickle=False):

        if to_pickle: # for training purpose
            
            with open('pickle_files/{}_frames.pickle'.format(path),'wb') as f:
                pickle.dump(self.frames,f)
            
            with open('pickle_files/{}_boxes.pickle'.format(path),'wb') as f:
                pickle.dump(self.boxes,f)
        
        else: # for viewing purpose
            
            width, height = self.frame_size[1], self.frame_size[0]
            code = cv2.VideoWriter_fourcc('M','J','P','G')
            out  = cv2.VideoWriter('{}.avi'.format(path), code, 5, (width, height), 0)
            
            for i, frame in enumerate(self.frames):
                out.write(frame)
    
            out.release()

    def _sample(self,img_size,img_angle):
        # create a variation of the image based on size and rotation angle
        def change_size(img, length): # width, height
            return cv2.resize(img, (length, length), interpolation=cv2.INTER_LINEAR)

        def change_angle(img, angle):
            rows,cols = img.shape
            M = cv2.getRotationMatrix2D((cols//2,rows//2),angle,1)
            return cv2.warpAffine(img,M,(cols,rows))

        img = change_size(self.image, img_size)
        img = change_angle(img, img_angle)

        return img

    def _background(self):
        img = np.random.choice(range(150,256), self.frame_size)
        return img.astype(np.uint8)

    def _combine(self, bg, target, position):
        # attach target onto the background at the position assigned
        x1, y1, x2, y2 = position # width, height
        roi = bg[y1:y2, x1:x2]
        mask_inv = cv2.bitwise_not(target)
        bg2 = cv2.bitwise_and(roi,roi,mask = mask_inv)
        dst = cv2.add(bg2,target)
        bg[y1:y2, x1:x2] = dst
        return bg

    def _next_position(self, cur_pos, step): 
        x1, y1, x2, y2 = cur_pos 
        till_top = max(y1 - self.margin, 0)
        till_left = max(x1 - self.margin, 0)
        till_bottom = max(self.frame_size[0] - self.margin - y2, 0)  
        till_right = max(self.frame_size[1] - self.margin - x2, 0)
    
        # The image will move toward a direction based on hitorical trajectory
        # as long as it doesnt touch the border
        if 0 not in (till_top, till_left, till_bottom, till_right):
            total = sum(self.trajectory)
            p = [pp/total for pp in self.trajectory]
            move = np.random.choice(self.moves, 1, p = p)[0]
            self.trajectory[self.moves.index(move)] += 1
        
        # The image will be force to move inward once it touch the border
        else:
            self.trajectory = [1,1,1,1]
            if till_top == 0:
                move = 'down'
            elif till_left == 0:
                move = 'right'
            elif till_bottom == 0:
                move = 'up'
            elif till_right == 0:
                move = 'left'
                    
        if move == 'up':
            step = min(step, till_top)
            return (x1, y1-step, x2, y2-step)
        elif move == 'down':
            step = min(step, till_bottom)
            return (x1, y1+step, x2, y2+step)
        elif move == 'left':
            step = min(step, till_left)
            return (x1-step, y1, x2-step, y2)
        else:
            step = min(step, till_right)
            return (x1+step, y1, x2+step, y2)


def make_video_from_pickle(src_frame_path, src_boxes_path, output_path):

    with open('pickle_files/{}.pickle'.format(src_frame_path),'rb') as f:
        frames = pickle.load(f)
    
    with open('pickle_files/{}.pickle'.format(src_boxes_path),'rb') as f:
        boxes = pickle.load(f)

    height, width = frames[0].shape
    code = cv2.VideoWriter_fourcc('M','J','P','G')
    out  = cv2.VideoWriter('{}.avi'.format(output_path), code, 5, (width, height), 0) 

    for i, frame in enumerate(frames):

        pos    = boxes[i]
        x1, y1 = int(pos[0]*width), int(pos[1]*height)
        x2, y2 = int(pos[2]*width), int(pos[3]*height)                   
        result = cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,0), 2)
        out.write(frame)

    out.release()








