
import rospy
import rosbag
import os
import sys
import argparse


def remove_tf(inbag, outbag, frame_ids):
    print("   Processing input bagfile: ", inbag)
    print("  Writing to output bagfile: ", outbag)
    print("         Removing frame_ids: ", " ".join(frame_ids))

    outbag = rosbag.Bag(outbag, "w")
    for topic, msg, t in rosbag.Bag(inbag, "r").read_messages():
        if topic == "tf":
            new_transforms = []
            for transform in msg.transforms:
                if (
                    transform.header.frame_id in frame_ids
                    and transform.child_frame_id in frame_ids
                ):
                    new_transforms.append(transform)
            msg.transforms = new_transforms
        outbag.write(topic, msg, t)
    print("Closing output bagfile and exit...")
    outbag.flush()
    outbag.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="removes all transforms from the /tf topic that contain one of the given frame_ids in the header as parent or child."
    )
    parser.add_argument("-i", metavar="INPUT_BAGFILE", required=True, help="input bagfile")
    parser.add_argument("-o", metavar="OUTPUT_BAGFILE", required=True, help="output bagfile")
    parser.add_argument(
        "-f",
        metavar="FRAME_ID",
        required=True,
        help="frame_id(s) of the transforms to keep in /tf topic",
        nargs="+",
    )
    args = parser.parse_args()

    try:
        remove_tf(args.i, args.o, args.f)
    except Exception as e:
        import traceback

        traceback.print_exc()
