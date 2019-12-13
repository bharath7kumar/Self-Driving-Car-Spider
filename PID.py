import cv2
import numpy as np 

class pidcontrol:
	def __init__(self,image,lw=0,lb=0,rw=0,rb=0):
		self.img=image
		self.lw=lw
		self.lb=lb
		self.rw=rw
		self.rb=rb

	def speed(self,lw,rw):
		leftwheel=0
		rightwheel=0

		error=lw-rw

		if(error<-3000):
			rightwheel=100*(1-(abs(error)/92601))
			leftwheel=100
		elif(error>=3000):
			rightwheel=100
			leftwheel=100*(1-(abs(error)/92601))

		elif((error>-3000)&(error<3000)):
			rightwheel=100
			leftwheel=100


		print("Leftwheel:",leftwheel)
		print("Rightwheel:",rightwheel)


		return error



	def nopixels(self):
		whitex,whitey=np.nonzero(self.img)
		whitey[whitey<320] =0
		whitey[whitey>=320]=1
		whitey=list(whitey)
		self.lw=whitey.count(0)
		self.rw=whitey.count(1)
		#print(self.lw,self.rw)
		error=self.speed(self.lw,self.rw)
		return error





	def vel(self,theta,centerx,cx,centery,x):
		leftwheel=0
		rightwheel=0

		maxspeed=55
		minspeed=-55

		pt1=np.array((0,centery))
		pt2=np.array((centerx,0))
		pt3=np.array((x,centery))

		hyp1=np.linalg.norm(pt1-pt2)
		hyp2=np.linalg.norm(pt3-pt2)

		mintheta=-90
		maxtheta=90
		if(theta>0):
			rightwheel= (theta/maxtheta)*maxspeed +10
			leftwheel=  (theta/maxtheta)*minspeed +10
		elif(theta<0):
			leftwheel= (theta/mintheta)*maxspeed +10
			rightwheel=(theta/mintheta)*minspeed +10
		else:
			leftwheel=10
			rightwheel=10
		print("Leftwheel:",leftwheel)
		print("wheel:",rightwheel)
		return(leftwheel,rightwheel)

	def contourarea(self,img2):

		im2,contours,hierarchy=cv2.findContours(self.img,1,2)
		#print(contours)
		for indx in range(0,len(contours)):
			for j in range(1,len(contours)):
				area=cv2.contourArea(contours[indx])
				area2=cv2.contourArea(contours[j])
				if(area2>area):
					temp=contours[indx]
					contours[indx]=contours[j]
					contours[j]=temp

		M1=cv2.moments(contours[0])
		cx1=int(M1['m10']/M1['m00'])  #centroid
		cy1=int(M1['m01']/M1['m00'])
		cx=cx1
		cy=cy1
		
		try:
			M2=cv2.moments(contours[1])
			cx2=int(M2['m10']/M2['m00'])
			cy2=int(M2['m01']/M2['m00'])

			cv2.drawContours(img2,contours,1,(200,200,200),3)

			if(cy1<0.2*self.img.shape[0]):
				cx=cx2
				cy=cy2   
		
		except:
			cx=cx1
			cy=cy1
  
	    

		centerx=int(img2.shape[1]/2)
		cv2.line(img2,(cx,cy),(centerx,cy),(255,0,0),2)
		
		distance= cx-centerx #distance between center and centroid
		cv2.line(img2,(centerx,0),(centerx,img2.shape[0]),(0,0,255),1)
		
		cv2.line(img2,(centerx,img2.shape[0]),(cx,cy),(200,200,200),2)
		pt1=np.array((centerx,img2.shape[0]))
		pt2=np.array((cx,cy))
		
		hyp= np.linalg.norm(pt1-pt2) #hypotenuse
		theta=np.arcsin((distance)/(hyp))*180/np.pi #angle
		print(theta)
		
		#self.vel(theta,centerx,cx,cy,img2.shape[1])
		

		cv2.drawContours(img2,contours,0,(0,255,0),3)	
		cv2.imshow('contourimage',img2)
		return (cx,cy)


