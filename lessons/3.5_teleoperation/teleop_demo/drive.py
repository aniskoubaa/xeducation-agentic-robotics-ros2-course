"""Lesson 3.5 - drive the arm by hand, the way you would in the lab.

WHAT  Runs the lab's real keyboard teleop (starter/01_teleop.py) and presses a
      short, fixed sequence of keys into it, so the demo drives the same way in
      every take.  The keys, the node and the published commands are all real -
      only the fingers are scripted.

WHY   This is what "teleoperation" means, at the lowest level: every key press
      publishes ONE absolute joint target, and the arm goes there.  Nobody
      writes the motion; you show it.  Count the presses and you also learn why
      collecting fifty episodes this way takes an hour.

    python3 drive.py                 # 1 1 1 w w c o x - swing, lower, grip
    python3 drive.py "111ccooox" 0.8 # any key string, any gap in seconds
"""
import os
import pty
import select
import subprocess
import sys
import time

KEYS = sys.argv[1] if len(sys.argv) > 1 else '111wwcox'
GAP = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0   # seconds between presses

# A pseudo-terminal, because 01_teleop.py reads single keys in raw tty mode:
# a plain pipe has no termios and the node would refuse to start.
main_fd, child_fd = pty.openpty()
teleop = subprocess.Popen(['ros2', 'run', 'raise2026_labs', '01_teleop.py'],
                          stdin=child_fd, stdout=child_fd, stderr=child_fd,
                          close_fds=True)
os.close(child_fd)


def show(seconds):
    """Echo whatever the teleop node prints, for this long. False = it exited."""
    end = time.time() + seconds
    while time.time() < end:
        if select.select([main_fd], [], [], 0.05)[0]:
            try:
                out = os.read(main_fd, 4096)
            except OSError:
                return False
            if not out:
                return False
            sys.stdout.write(out.decode('utf-8', 'replace'))
            sys.stdout.flush()
    return True


show(3.0)                            # let the node print its key map
for key in KEYS:
    if key != ' ':                   # a space in KEYS is just a pause
        os.write(main_fd, key.encode())
    if not show(GAP):
        break
if not KEYS.rstrip().endswith('x'):  # 'x' quits; without it the node runs forever
    os.write(main_fd, b'x')
show(2.0)
try:
    teleop.wait(timeout=10)
except subprocess.TimeoutExpired:    # never let a stuck node print a traceback
    teleop.terminate()
    teleop.wait(timeout=5)
