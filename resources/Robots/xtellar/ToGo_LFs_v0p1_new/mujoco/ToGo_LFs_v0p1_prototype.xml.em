<mujoco model='XtellarRobotics/ToGo_LFs_v0p1_prototype'>
    <compiler meshdir='../meshes' />

    <default>
        <default class="vis">
            <geom
                type="mesh" rgba="@_main_rgba[0] @_main_rgba[1] @_main_rgba[2] @_main_rgba[3]"
                group="1"
                contype="0" conaffinity="0" density="0"
            />
        </default>
        <default class="vis_rubber">
            <geom
                type="mesh" rgba="0.149 0.149 0.149 @_main_rgba[3]"
                group="1"
                contype="0" conaffinity="0" density="0"
            />
        </default>
        <default class="col">
            <geom
                type="mesh" rgba="@_main_rgba[0] @_main_rgba[1] @_main_rgba[2] @_main_rgba[3]"
                group="3"
                contype="1" conaffinity="1" density="0"
                solref="0.005 1" solimp="0.95 0.99 0.001 0.5 2"
            />
        </default>
        <default class="col_rubber">
            <geom
                type="mesh" rgba="0.149 0.149 0.149 @_main_rgba[3]"
                group="3"
                contype="1" conaffinity="1" density="0"
                solref="0.005 1" solimp="0.95 0.99 0.001 0.5 2"
                condim="6" friction="1.5 0.02 0.001"
            />
        </default>
    </default>

    <asset>
        <mesh name='vis_m0_torso'   file='visual/m0_torso.stl'      scale='0.001 0.001 0.001' />
        <!-- front left leg -->
        <mesh name='vis_mfl1_hipr'  file='visual/mfl1_hipr.stl'     scale='0.001 0.001 0.001' />
        <mesh name='vis_mfl2_hipp'  file='visual/mfl2_hipp.stl'     scale='0.001 0.001 0.001' />
        <mesh name='vis_mfl3_knee'  file='visual/mfl3_knee.stl'     scale='0.001 0.001 0.001' />
        <mesh name='vis_mfl4_foot'  file='visual/mfl4_foot.stl'     scale='0.001 0.001 0.001' />
        <!-- front right leg -->
        <mesh name='vis_mfr1_hipr'  file='visual/mfr1_hipr.stl'     scale='0.001 0.001 0.001' />
        <mesh name='vis_mfr2_hipp'  file='visual/mfr2_hipp.stl'     scale='0.001 0.001 0.001' />
        <mesh name='vis_mfr3_knee'  file='visual/mfr3_knee.stl'     scale='0.001 0.001 0.001' />
        <mesh name='vis_mfr4_foot'  file='visual/mfr4_foot.stl'     scale='0.001 0.001 0.001' />
        <!-- rear left leg -->
        <mesh name='vis_mrl1_hipr'  file='visual/mrl1_hipr.stl'     scale='0.001 0.001 0.001' />
        <mesh name='vis_mrl2_hipp'  file='visual/mrl2_hipp.stl'     scale='0.001 0.001 0.001' />
        <mesh name='vis_mrl3_knee'  file='visual/mrl3_knee.stl'     scale='0.001 0.001 0.001' />
        <mesh name='vis_mrl4_foot'  file='visual/mrl4_foot.stl'     scale='0.001 0.001 0.001' />
        <!-- rear right leg -->
        <mesh name='vis_mrr1_hipr'  file='visual/mrr1_hipr.stl'     scale='0.001 0.001 0.001' />
        <mesh name='vis_mrr2_hipp'  file='visual/mrr2_hipp.stl'     scale='0.001 0.001 0.001' />
        <mesh name='vis_mrr3_knee'  file='visual/mrr3_knee.stl'     scale='0.001 0.001 0.001' />
        <mesh name='vis_mrr4_foot'  file='visual/mrr4_foot.stl'     scale='0.001 0.001 0.001' />

        <mesh name='col_m0_torso'   file='collision/m0_torso.stl'   scale='0.001 0.001 0.001' />
        <!-- front left leg -->
        <mesh name='col_mfl1_hipr'  file='collision/mfl1_hipr.stl'  scale='0.001 0.001 0.001' />
        <mesh name='col_mfl2_hipp'  file='collision/mfl2_hipp.stl'  scale='0.001 0.001 0.001' />
        <mesh name='col_mfl3_knee'  file='collision/mfl3_knee.stl'  scale='0.001 0.001 0.001' />
        <mesh name='col_mfl4_foot'  file='collision/mfl4_foot.stl'  scale='0.001 0.001 0.001' />
        <!-- front right leg -->
        <mesh name='col_mfr1_hipr'  file='collision/mfr1_hipr.stl'  scale='0.001 0.001 0.001' />
        <mesh name='col_mfr2_hipp'  file='collision/mfr2_hipp.stl'  scale='0.001 0.001 0.001' />
        <mesh name='col_mfr3_knee'  file='collision/mfr3_knee.stl'  scale='0.001 0.001 0.001' />
        <mesh name='col_mfr4_foot'  file='collision/mfr4_foot.stl'  scale='0.001 0.001 0.001' />
        <!-- rear left leg -->
        <mesh name='col_mrl1_hipr'  file='collision/mrl1_hipr.stl'  scale='0.001 0.001 0.001' />
        <mesh name='col_mrl2_hipp'  file='collision/mrl2_hipp.stl'  scale='0.001 0.001 0.001' />
        <mesh name='col_mrl3_knee'  file='collision/mrl3_knee.stl'  scale='0.001 0.001 0.001' />
        <mesh name='col_mrl4_foot'  file='collision/mrl4_foot.stl'  scale='0.001 0.001 0.001' />
        <!-- rear right leg -->
        <mesh name='col_mrr1_hipr'  file='collision/mrr1_hipr.stl'  scale='0.001 0.001 0.001' />
        <mesh name='col_mrr2_hipp'  file='collision/mrr2_hipp.stl'  scale='0.001 0.001 0.001' />
        <mesh name='col_mrr3_knee'  file='collision/mrr3_knee.stl'  scale='0.001 0.001 0.001' />
        <mesh name='col_mrr4_foot'  file='collision/mrr4_foot.stl'  scale='0.001 0.001 0.001' />
    </asset>

    <worldbody>
        @(taghead_body(link_name='F_base', indent=2))
            @(taghead_body(link_name='L0_torso', indent=3))
                @(tag_inertial(link_name='L0_torso', indent=4))
                <geom class="vis" mesh="vis_m0_torso" />
                <geom class="col" mesh="col_m0_torso" />

                @(tag_site(name='imu', ref_joint='J_L0_torso_F_imu'))

                <!-- front left leg -->
                @(taghead_body(link_name='Lfl1_hipr'))
                    @(tag_joint(joint_name='Jfl1_hipr', indent=5))
                    @(tag_inertial(link_name='Lfl1_hipr'))
                    <geom class="vis" mesh="vis_mfl1_hipr" />
                    <geom class="col" mesh="col_mfl1_hipr" />

                    @(taghead_body(link_name='Lfl2_hipp'))
                        @(tag_joint(joint_name='Jfl2_hipp'))
                        @(tag_inertial(link_name='Lfl2_hipp'))
                        <geom class="vis" mesh="vis_mfl2_hipp" />
                        <geom class="col" mesh="col_mfl2_hipp" />

                        @(taghead_body(link_name='Lfl3_knee'))
                            @(tag_joint(joint_name='Jfl3_knee'))
                            @(tag_inertial(link_name='Lfl3_knee'))
                            <geom class="vis" mesh="vis_mfl3_knee" />
                            <geom class="col" mesh="col_mfl3_knee" />

                            @(taghead_body(link_name='Lfl4_foot'))
                                @(tag_inertial(link_name='Lfl4_foot'))
                                <geom class="vis_rubber" mesh="vis_mfl4_foot" />
                                <geom class="col_rubber" mesh="col_mfl4_foot" />
                            </body>
                        </body>
                    </body>
                </body>

                <!-- front right leg -->
                @(taghead_body(link_name='Lfr1_hipr', indent=4))
                    @(tag_joint(joint_name='Jfr1_hipr', indent=5))
                    @(tag_inertial(link_name='Lfr1_hipr', indent=5))
                    <geom class="vis" mesh="vis_mfr1_hipr" />
                    <geom class="col" mesh="col_mfr1_hipr" />

                    @(taghead_body(link_name='Lfr2_hipp'))
                        @(tag_joint(joint_name='Jfr2_hipp'))
                        @(tag_inertial(link_name='Lfr2_hipp'))
                        <geom class="vis" mesh="vis_mfr2_hipp" />
                        <geom class="col" mesh="col_mfr2_hipp" />

                        @(taghead_body(link_name='Lfr3_knee'))
                            @(tag_joint(joint_name='Jfr3_knee'))
                            @(tag_inertial(link_name='Lfr3_knee'))
                            <geom class="vis" mesh="vis_mfr3_knee" />
                            <geom class="col" mesh="col_mfr3_knee" />

                            @(taghead_body(link_name='Lfr4_foot'))
                                @(tag_inertial(link_name='Lfr4_foot'))
                                <geom class="vis_rubber" mesh="vis_mfr4_foot" />
                                <geom class="col_rubber" mesh="col_mfr4_foot" />
                            </body>
                        </body>
                    </body>
                </body>

                <!-- rear left leg -->
                @(taghead_body(link_name='Lrl1_hipr', indent=4))
                    @(tag_joint(joint_name='Jrl1_hipr', indent=5))
                    @(tag_inertial(link_name='Lrl1_hipr', indent=5))
                    <geom class="vis" mesh="vis_mrl1_hipr" />
                    <geom class="col" mesh="col_mrl1_hipr" />

                    @(taghead_body(link_name='Lrl2_hipp'))
                        @(tag_joint(joint_name='Jrl2_hipp'))
                        @(tag_inertial(link_name='Lrl2_hipp'))
                        <geom class="vis" mesh="vis_mrl2_hipp" />
                        <geom class="col" mesh="col_mrl2_hipp" />

                        @(taghead_body(link_name='Lrl3_knee'))
                            @(tag_joint(joint_name='Jrl3_knee'))
                            @(tag_inertial(link_name='Lrl3_knee'))
                            <geom class="vis" mesh="vis_mrl3_knee" />
                            <geom class="col" mesh="col_mrl3_knee" />

                            @(taghead_body(link_name='Lrl4_foot'))
                                @(tag_inertial(link_name='Lrl4_foot'))
                                <geom class="vis_rubber" mesh="vis_mrl4_foot" />
                                <geom class="col_rubber" mesh="col_mrl4_foot" />
                            </body>
                        </body>
                    </body>
                </body>

                <!-- rear right leg -->
                @(taghead_body(link_name='Lrr1_hipr', indent=4))
                    @(tag_joint(joint_name='Jrr1_hipr', indent=5))
                    @(tag_inertial(link_name='Lrr1_hipr', indent=5))
                    <geom class="vis" mesh="vis_mrr1_hipr" />
                    <geom class="col" mesh="col_mrr1_hipr" />

                    @(taghead_body(link_name='Lrr2_hipp'))
                        @(tag_joint(joint_name='Jrr2_hipp'))
                        @(tag_inertial(link_name='Lrr2_hipp'))
                        <geom class="vis" mesh="vis_mrr2_hipp" />
                        <geom class="col" mesh="col_mrr2_hipp" />

                        @(taghead_body(link_name='Lrr3_knee'))
                            @(tag_joint(joint_name='Jrr3_knee'))
                            @(tag_inertial(link_name='Lrr3_knee'))
                            <geom class="vis" mesh="vis_mrr3_knee" />
                            <geom class="col" mesh="col_mrr3_knee" />

                            @(taghead_body(link_name='Lrr4_foot'))
                                @(tag_inertial(link_name='Lrr4_foot'))
                                <geom class="vis_rubber" mesh="vis_mrr4_foot" />
                                <geom class="col_rubber" mesh="col_mrr4_foot" />
                            </body>
                        </body>
                    </body>
                </body>

            </body>
        </body>
    </worldbody>

    <contact>
        <!-- front left leg -->
        <exclude body1="L0_torso"   body2="Lfl1_hipr" />
        <exclude body1="L0_torso"   body2="Lfl2_hipp" />
        <exclude body1="Lfl1_hipr"  body2="Lfl2_hipp" />
        <exclude body1="Lfl2_hipp"  body2="Lfl3_knee" />
        <!-- front right leg -->
        <exclude body1="L0_torso"   body2="Lfr1_hipr" />
        <exclude body1="L0_torso"   body2="Lfr2_hipp" />
        <exclude body1="Lfr1_hipr"  body2="Lfr2_hipp" />
        <exclude body1="Lfr2_hipp"  body2="Lfr3_knee" />
        <!-- rear left leg -->
        <exclude body1="L0_torso"   body2="Lrl1_hipr" />
        <exclude body1="L0_torso"   body2="Lrl2_hipp" />
        <exclude body1="Lrl1_hipr"  body2="Lrl2_hipp" />
        <exclude body1="Lrl2_hipp"  body2="Lrl3_knee" />
        <!-- rear right leg -->
        <exclude body1="L0_torso"   body2="Lrr1_hipr" />
        <exclude body1="L0_torso"   body2="Lrr2_hipp" />
        <exclude body1="Lrr1_hipr"  body2="Lrr2_hipp" />
        <exclude body1="Lrr2_hipp"  body2="Lrr3_knee" />
    </contact>

    <actuator>
        <!-- front left leg -->
        @(tag_motor(actuator_name='Rfl1_hipr', indent=2))
        @(tag_motor(actuator_name='Rfl2_hipp', indent=2))
        @(tag_motor(actuator_name='Rfl3_knee', indent=2))
        <!-- front right leg -->
        @(tag_motor(actuator_name='Rfr1_hipr', indent=2))
        @(tag_motor(actuator_name='Rfr2_hipp', indent=2))
        @(tag_motor(actuator_name='Rfr3_knee', indent=2))
        <!-- rear left leg -->
        @(tag_motor(actuator_name='Rrl1_hipr', indent=2))
        @(tag_motor(actuator_name='Rrl2_hipp', indent=2))
        @(tag_motor(actuator_name='Rrl3_knee', indent=2))
        <!-- rear right leg -->
        @(tag_motor(actuator_name='Rrr1_hipr', indent=2))
        @(tag_motor(actuator_name='Rrr2_hipp', indent=2))
        @(tag_motor(actuator_name='Rrr3_knee', indent=2))
    </actuator>

    <sensor>
        <!-- joint position `q_J` -->

        <!--- front left leg -->
        <jointpos name="q_Jfl1_hipr" joint="Jfl1_hipr" noise='@_jpos_noise' />
        <jointpos name="q_Jfl2_hipp" joint="Jfl2_hipp" noise='@_jpos_noise' />
        <jointpos name="q_Jfl3_knee" joint="Jfl3_knee" noise='@_jpos_noise' />
        <!--- front right leg -->
        <jointpos name="q_Jfr1_hipr" joint="Jfr1_hipr" noise='@_jpos_noise' />
        <jointpos name="q_Jfr2_hipp" joint="Jfr2_hipp" noise='@_jpos_noise' />
        <jointpos name="q_Jfr3_knee" joint="Jfr3_knee" noise='@_jpos_noise' />
        <!--- rear left leg -->
        <jointpos name="q_Jrl1_hipr" joint="Jrl1_hipr" noise='@_jpos_noise' />
        <jointpos name="q_Jrl2_hipp" joint="Jrl2_hipp" noise='@_jpos_noise' />
        <jointpos name="q_Jrl3_knee" joint="Jrl3_knee" noise='@_jpos_noise' />
        <!--- rear right leg -->
        <jointpos name="q_Jrr1_hipr" joint="Jrr1_hipr" noise='@_jpos_noise' />
        <jointpos name="q_Jrr2_hipp" joint="Jrr2_hipp" noise='@_jpos_noise' />
        <jointpos name="q_Jrr3_knee" joint="Jrr3_knee" noise='@_jpos_noise' />

        <!-- joint velocity `dq_J` -->

        <!--- front left leg -->
        <jointvel name="dq_Jfl1_hipr" joint="Jfl1_hipr" noise='@_jvel_noise' />
        <jointvel name="dq_Jfl2_hipp" joint="Jfl2_hipp" noise='@_jvel_noise' />
        <jointvel name="dq_Jfl3_knee" joint="Jfl3_knee" noise='@_jvel_noise' />
        <!--- front right leg -->
        <jointvel name="dq_Jfr1_hipr" joint="Jfr1_hipr" noise='@_jvel_noise' />
        <jointvel name="dq_Jfr2_hipp" joint="Jfr2_hipp" noise='@_jvel_noise' />
        <jointvel name="dq_Jfr3_knee" joint="Jfr3_knee" noise='@_jvel_noise' />
        <!--- rear left leg -->
        <jointvel name="dq_Jrl1_hipr" joint="Jrl1_hipr" noise='@_jvel_noise' />
        <jointvel name="dq_Jrl2_hipp" joint="Jrl2_hipp" noise='@_jvel_noise' />
        <jointvel name="dq_Jrl3_knee" joint="Jrl3_knee" noise='@_jvel_noise' />
        <!--- rear right leg -->
        <jointvel name="dq_Jrr1_hipr" joint="Jrr1_hipr" noise='@_jvel_noise' />
        <jointvel name="dq_Jrr2_hipp" joint="Jrr2_hipp" noise='@_jvel_noise' />
        <jointvel name="dq_Jrr3_knee" joint="Jrr3_knee" noise='@_jvel_noise' />

        <!-- joint torque `tau_J` -->

        <!--- front left leg -->
        <jointactuatorfrc name="tau_Jfl1_hipr" joint="Jfl1_hipr" noise='@_jtor_noise' />
        <jointactuatorfrc name="tau_Jfl2_hipp" joint="Jfl2_hipp" noise='@_jtor_noise' />
        <jointactuatorfrc name="tau_Jfl3_knee" joint="Jfl3_knee" noise='@_jtor_noise' />
        <!--- front right leg -->
        <jointactuatorfrc name="tau_Jfr1_hipr" joint="Jfr1_hipr" noise='@_jtor_noise' />
        <jointactuatorfrc name="tau_Jfr2_hipp" joint="Jfr2_hipp" noise='@_jtor_noise' />
        <jointactuatorfrc name="tau_Jfr3_knee" joint="Jfr3_knee" noise='@_jtor_noise' />
        <!--- rear left leg -->
        <jointactuatorfrc name="tau_Jrl1_hipr" joint="Jrl1_hipr" noise='@_jtor_noise' />
        <jointactuatorfrc name="tau_Jrl2_hipp" joint="Jrl2_hipp" noise='@_jtor_noise' />
        <jointactuatorfrc name="tau_Jrl3_knee" joint="Jrl3_knee" noise='@_jtor_noise' />
        <!--- rear right leg -->
        <jointactuatorfrc name="tau_Jrr1_hipr" joint="Jrr1_hipr" noise='@_jtor_noise' />
        <jointactuatorfrc name="tau_Jrr2_hipp" joint="Jrr2_hipp" noise='@_jtor_noise' />
        <jointactuatorfrc name="tau_Jrr3_knee" joint="Jrr3_knee" noise='@_jtor_noise' />

        <!-- IMU -->
        <gyro           name="imu_gyro" site="imu" noise="@_gyro_noise" />
        <accelerometer  name="imu_accl" site="imu" noise="@_accl_noise" />
        <framequat      name="imu_quat" objtype="site" objname="imu" noise="@_orie_noise" />
        <framepos       name="gt_pos"   objtype="site" objname="imu" />
        <framelinvel    name="gt_vel"   objtype="site" objname="imu" />
    </sensor>
</mujoco>
