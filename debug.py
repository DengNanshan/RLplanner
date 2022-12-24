
    def publish_planningmsg_trajectory(self, trajectory):
                # print("trajectory.trajectory",trajectory.trajectory)
                if not self.localization_received:
                    self.logger.warning(
                        "localization not received yet when publish_planningmsg")
                    return

                planningdata = planning_pb2.ADCTrajectory()
                now = cyber_time.Time.now().to_sec()
                planningdata.header.timestamp_sec = now
                planningdata.header.module_name = "planning"
                planningdata.header.sequence_num = self.sequence_num
                self.sequence_num = self.sequence_num + 1


                planningdata.total_path_length = self.data['s'][self.end] - \
                    self.data['s'][self.start]
                self.logger.info("total number of planning data point: %d" %
                                (self.end - self.start))
                planningdata.total_path_time = self.data['time'][self.end] - \
                    self.data['time'][self.start]
                planningdata.gear = 1
                planningdata.engage_advice.advice = \
                    drive_state_pb2.EngageAdvice.READY_TO_ENGAGE

                for i in range(len(trajectory.trajectory)-1):
                    adc_point = pnc_point_pb2.TrajectoryPoint()
                    adc_point.path_point.x = trajectory.trajectory[i][0]
                    adc_point.path_point.y = trajectory.trajectory[i][1]
                    adc_point.path_point.z = 0
                    adc_point.v =  trajectory.trajectory[i][3]
                    adc_point.a =  trajectory.trajectory[i][5]
                    adc_point.path_point.kappa = 0
                    adc_point.path_point.dkappa =0
                    adc_point.path_point.theta = 0# trajectory.trajectory[i][2]
                    adc_point.path_point.s = trajectory.trajectory[i][4]


                    time_diff = self.data['time'][i] - \
                        self.data['time'][0]

                    adc_point.relative_time = time_diff  - (
                        now - self.starttime)

                    planningdata.trajectory_point.extend([adc_point])

                planningdata.estop.is_estop = False

                self.planning_pub.write(planningdata)
                self.logger.debug("Generated Planning Sequence: "
                                + str(self.sequence_num - 1))
