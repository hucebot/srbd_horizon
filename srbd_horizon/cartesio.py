from geometry_msgs.msg import PoseStamped, TwistStamped
import rospy

class cartesIO_struct:
    def __init__(self, distal_link, base_link="world"):
        self.pose_publisher = rospy.Publisher(f"/cartesian/{distal_link}/reference", PoseStamped, queue_size=1)
        self.vel_publisher = rospy.Publisher(f"/cartesian/{distal_link}/velocity_reference", TwistStamped, queue_size=1)

        self.pose = PoseStamped()
        self.pose.pose.position.x = self.pose.pose.position.y = self.pose.pose.position.z = 0.
        self.pose.pose.orientation.x = self.pose.pose.orientation.y = self.pose.pose.orientation.z = 0.
        self.pose.pose.orientation.w = 1.
        self.pose.header.frame_id = "world"

        self.vel = TwistStamped()
        self.vel.twist.angular.x = self.vel.twist.angular.y = self.vel.twist.angular.z = 0.
        self.vel.header.frame_id = "world"

    def setPosition(self, p):
        self.pose.pose.position.x = p[0]
        self.pose.pose.position.y = p[1]
        self.pose.pose.position.z = p[2]

    def setOrientation(self, o):
        self.pose.pose.orientation.x = o[0]
        self.pose.pose.orientation.y = o[1]
        self.pose.pose.orientation.z = o[2]

    def setLinearVelocity(self, v):
        self.vel.twist.linear.x = v[0]
        self.vel.twist.linear.y = v[1]
        self.vel.twist.linear.z = v[2]

    def setAngularVelocity(self, w):
        self.vel.twist.angular.x = w[0]
        self.vel.twist.angular.y = w[1]
        self.vel.twist.angular.z = w[2]

    def publish(self, t):
        self.pose.header.stamp = t
        self.vel.header.stamp = t
        self.pose_publisher.publish(self.pose)
        self.vel_publisher.publish(self.vel)



class cartesIO:
    def __init__(self, contact_frames):
        self.contacts = dict()
        for frame in contact_frames:
            self.contacts[frame] = cartesIO_struct(frame)

        self.com = cartesIO_struct("com")

        self.base_link = cartesIO_struct("base_link")

    #c is a dict {contact_frame: [contacts]}
    def publish(self, r, rdot, o, w,  c, cdot, t):

        self.com.setPosition(r)
        #self.com.setLinearVelocity(rdot)

        self.base_link.setOrientation(o)
        #self.base_link.setAngularVelocity(w)

        for frame in c:
            contact_list = c[frame]
            if len(contact_list) == 2: #line feet
                p0 = contact_list[0]
                p1 = contact_list[1]
                p = (p0 + p1)/2.
                self.contacts[frame].setPosition(p)

            #self.contacts[frame].setLinearVelocity(cdot[frame][0])

        self.com.publish(t)
        self.base_link.publish(t)
        for frame in c:
            self.contacts[frame].publish(t)
