import argparse
import time
import visdom
from enum import Enum

import numpy as np

from udacidrone import Drone
from udacidrone.connection import MavlinkConnection, WebSocketConnection  # noqa: F401
from udacidrone.messaging import MsgID


class States(Enum):
    MANUAL = 0
    ARMING = 1
    TAKEOFF = 2
    WAYPOINT = 3
    LANDING = 4
    DISARMING = 5


class BackyardFlyer(Drone):

    def __init__(self, connection, edge, altitude, delta):
        super().__init__(connection)
        self.v = visdom.Visdom()
        assert self.v.check_connection()
        self.startLocation = np.array([0.0, 0.0, 0.0])
        self.target_position = np.array([0.0, 0.0, 0.0])

        # Plot NE
        ne = np.array([0, 0]).reshape(-1, 2)
        self.ne_plot = self.v.scatter(ne,
                                      opts=dict(
            title="Local position (north, east)",
            xlabel='North',
            ylabel='East'
        ))

        # Plot D
        d = np.array([-self.local_position[2]])
        self.t = 1
        self.d_plot = self.v.line(d, X=np.array([self.t]), opts=dict(
            title="Altitude (meters)",
            xlabel='Timestep',
            ylabel='Down'
        ))

        self._edge = edge
        self._targetAltitude = altitude
        self._delta = delta
        self.all_waypoints = self.calculate_box()
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL

        #  Register all my callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.update_d_plot_callback)
        self.register_callback(MsgID.LOCAL_POSITION, self.update_ne_plot_callback)
        #  end of my callbacks
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def calculate_box(self):
        """
        1. Return waypoints to fly a box
        """
        edge = self._edge
        altitude = self._targetAltitude
        square = [np.array([edge, 0, altitude]),
                np.array([edge, edge, altitude]),
                np.array([0, edge, altitude]),
                np.array([0, 0, altitude])]
        square.reverse()
        return square

    def update_ne_plot_callback(self):
        if self.flight_state == States.TAKEOFF:
            ne = np.array([self.local_position[0]-self.startLocation[0],
                           self.local_position[1]-self.startLocation[1]]).reshape(-1, 2)
            self.v.scatter(ne, win=self.ne_plot, update='append')

    def update_d_plot_callback(self):
        d = np.array([-self.local_position[2]])
        # update timestep
        self.t += 1
        self.v.line(d, X=np.array([self.t]), win=self.d_plot, update='append')

    def local_position_callback(self):
        """
        This triggers when `MsgID.LOCAL_POSITION` is received and self.local_position contains new data
        """
        north, east, altitude = self.target_position + self.startLocation
        if self.flight_state == States.TAKEOFF:
            heading = 0
            self.cmd_position(north, east, altitude, heading)
            if abs(self.local_position[0] - north) < self._delta and abs(self.local_position[1] - east) < self._delta:
                if self.all_waypoints:
                    self.target_position = self.all_waypoints.pop()

    def velocity_callback(self):
        """
        This triggers when `MsgID.LOCAL_VELOCITY` is received and self.local_velocity contains new data
        """
        if self.flight_state == States.LANDING:
            self.disarming_transition()

    def state_callback(self):
        """

        This triggers when `MsgID.STATE` is received and self.armed and self.guided contain new data
        """
        if self.flight_state == States.MANUAL:
            self.arming_transition()
        elif self.flight_state == States.ARMING:
            self.takeoff_transition()
        elif self.flight_state == States.TAKEOFF:
            self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            self.landing_transition()
        elif self.flight_state == States.LANDING:
            self.disarming_transition()
        elif self.flight_state == States.DISARMING:
            self.manual_transition()

    def arming_transition(self):
        """
        1. Take control of the drone
        2. Pass an arming command
        3. Set the home location to current position
        4. Transition to the ARMING state
        """
        print("arming transition")
        self.take_control()
        self.arm()
        self.set_home_position(0, 0, 0)
        self.flight_state = States.ARMING

    def takeoff_transition(self):
        """
        1. Set target_position altitude to 3.0m
        2. Command a takeoff to 3.0m
        3. Transition to the TAKEOFF state
        """
        print("takeoff transition")
        self.takeoff(self._targetAltitude)
        if self.flight_state == States.ARMING and abs(self.local_position[2] + self._targetAltitude) < self._delta:
            self.startLocation[0] = self.local_position[0]
            self.startLocation[1] = self.local_position[1]
            self.flight_state = States.TAKEOFF

    def waypoint_transition(self):
        """
        1. Command the next waypoint position
        2. Transition to WAYPOINT state
        """
        if self.flight_state == States.TAKEOFF and len(self.all_waypoints) == 0 \
                and abs(self.local_position[0] - self.startLocation[0]) < self._delta \
                and abs(self.local_position[1] - self.startLocation[1]) < self._delta:
            self.flight_state = States.WAYPOINT 

    def landing_transition(self):
        """
        1. Command the drone to land
        2. Transition to the LANDING state
        """
        print("landing transition")
        self.land()
        if abs(self.local_position[2]) < self._delta:
            self.flight_state = States.LANDING

    def disarming_transition(self):
        """
        1. Command the drone to disarm
        2. Transition to the DISARMING state
        """
        print("disarm transition")
        self.disarm()
        self.flight_state = States.DISARMING

    def manual_transition(self):
        """This method is provided
        
        1. Release control of the drone
        2. Stop the connection (and telemetry log)
        3. End the mission
        4. Transition to the MANUAL state
        """
        print("manual transition")

        self.release_control()
        self.stop()
        self.in_mission = False
        self.flight_state = States.MANUAL

    def start(self):
        """This method is provided
        
        1. Open a log file
        2. Start the drone connection
        3. Close the log file
        """
        print("Creating log file")
        self.start_log("Logs", "MyNavLog.txt")
        print("starting connection")
        self.connection.start()
        print("Closing log file")
        self.stop_log()


if __name__ == "__main__":
    """
    Sample usage: 
    step 1:  $ python -m visdom.server 
    step 2: open a new terminal 
    $ python backyard_flyer.py --edge 250 --altitude 100 --precision 0.3
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")

    parser.add_argument('--edge', type=int, default=50, help='Length of the square edge')
    parser.add_argument('--altitude', type=int, default=40, help='Altitude of the flying')
    parser.add_argument('--precision', type=float, default=0.2, help='Precision to meet targets')
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), threaded=False, PX4=False)
    #conn = WebSocketConnection('ws://{0}:{1}'.format(args.host, args.port))
    squareEdge, altitude, delta = args.edge, args.altitude, args.precision
    drone = BackyardFlyer(conn, squareEdge, altitude, delta)
    time.sleep(2)
    print('starting drone...')
    drone.start()
