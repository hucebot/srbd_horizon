from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, WrenchStamped
import rospy

def publishPointTrj(points, t, name, frame, color = [0.7, 0.7, 0.7]):
    marker = Marker()
    marker.header.frame_id = frame
    marker.header.stamp = t
    marker.ns = "SRBD"
    marker.id = 1000
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD

    for k in range(0, points.shape[1]):
        p = Point()
        p.x = points[0, k]
        p.y = points[1, k]
        p.z = points[2, k]
        marker.points.append(p)

    marker.color.a = 1.
    marker.scale.x = 0.005
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]

    pub = rospy.Publisher(name + "_trj", Marker, queue_size=10).publish(marker)

def publishContactForce(t, f, frame):
    f_msg = WrenchStamped()
    f_msg.header.stamp = t
    f_msg.header.frame_id = frame
    f_msg.wrench.force.x = f[0]
    f_msg.wrench.force.y = f[1]
    f_msg.wrench.force.z = f[2]
    f_msg.wrench.torque.x = f_msg.wrench.torque.y = f_msg.wrench.torque.z = 0.
    pub = rospy.Publisher('f' + frame, WrenchStamped, queue_size=10).publish(f_msg)

def SRBDViewer(I, base_frame, t, number_of_contacts):
    marker = Marker()
    marker.header.frame_id = base_frame
    marker.header.stamp = t
    marker.ns = "SRBD"
    marker.id = 0
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = marker.pose.position.y = marker.pose.position.z = 0.
    marker.pose.orientation.x = marker.pose.orientation.y = marker.pose.orientation.z = 0.
    marker.pose.orientation.w = 1.
    a = I[0,0] + I[1,1] + I[2,2]
    marker.scale.x = 0.5*(I[2,2] + I[1,1])/a
    marker.scale.y = 0.5*(I[2,2] + I[0,0])/a
    marker.scale.z = 0.5*(I[0,0] + I[1,1])/a
    marker.color.a = 0.8
    marker.color.r = marker.color.g = marker.color.b = 0.7

    pub = rospy.Publisher('box', Marker, queue_size=10).publish(marker)

    marker_array = MarkerArray()
    for i in range(0, number_of_contacts):
        m = Marker()
        m.header.frame_id = "c" + str(i)
        m.header.stamp = t
        m.ns = "SRBD"
        m.id = i + 1
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = marker.pose.position.y = marker.pose.position.z = 0.
        m.pose.orientation.x = marker.pose.orientation.y = marker.pose.orientation.z = 0.
        m.pose.orientation.w = 1.
        m.scale.x = m.scale.y = m.scale.z = 0.04
        m.color.a = 0.8
        m.color.r = m.color.g = 0.0
        m.color.b = 1.0
        marker_array.markers.append(m)

    pub2 = rospy.Publisher('contacts', MarkerArray, queue_size=10).publish(marker_array)