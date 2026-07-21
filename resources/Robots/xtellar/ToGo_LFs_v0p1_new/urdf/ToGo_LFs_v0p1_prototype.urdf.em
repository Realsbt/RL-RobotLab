@[if _no_virtual_links]@
<robot name='XtellarRobotics/ToGo_LFs_v0p1_prototype_novlnk'>
@[else]@
<robot name='XtellarRobotics/ToGo_LFs_v0p1_prototype'>
@[end if]@

    <!-- mujoco tags -->
@[if _mujoco]@
    <mujoco>
        <compiler meshdir="../meshes" />
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
    </mujoco>
@[end if]@

    <material name="main">
        <color rgba="@_main_rgba[0] @_main_rgba[1] @_main_rgba[2] @_main_rgba[3]"/>
    </material>
    <material name="rubber_foot">
        <color rgba="0.149 0.149 0.149 @_main_rgba[3]"/>
    </material>

    <!-- link tags  -->

@[if not _no_virtual_links]@
    <link name='F_base' />
    <link name='F_imu' />
@[end if]@
    @(tag_link(link_name='L0_torso',    mesh_path='m0_torso.stl',   mesh_material='main', indent=1))
    <!-- front left leg -->
    @(tag_link(link_name='Lfl1_hipr',   mesh_path='mfl1_hipr.stl',  mesh_material='main', indent=1))
    @(tag_link(link_name='Lfl2_hipp',   mesh_path='mfl2_hipp.stl',  mesh_material='main', indent=1))
    @(tag_link(link_name='Lfl3_knee',   mesh_path='mfl3_knee.stl',  mesh_material='main', indent=1))
    @(tag_link(link_name='Lfl4_foot',   mesh_path='mfl4_foot.stl',  mesh_material='rubber_foot', indent=1))
    <!-- front right leg -->
    @(tag_link(link_name='Lfr1_hipr',   mesh_path='mfr1_hipr.stl',  mesh_material='main', indent=1))
    @(tag_link(link_name='Lfr2_hipp',   mesh_path='mfr2_hipp.stl',  mesh_material='main', indent=1))
    @(tag_link(link_name='Lfr3_knee',   mesh_path='mfr3_knee.stl',  mesh_material='main', indent=1))
    @(tag_link(link_name='Lfr4_foot',   mesh_path='mfr4_foot.stl',  mesh_material='rubber_foot', indent=1))
    <!-- rear left leg -->
    @(tag_link(link_name='Lrl1_hipr',   mesh_path='mrl1_hipr.stl',  mesh_material='main', indent=1))
    @(tag_link(link_name='Lrl2_hipp',   mesh_path='mrl2_hipp.stl',  mesh_material='main', indent=1))
    @(tag_link(link_name='Lrl3_knee',   mesh_path='mrl3_knee.stl',  mesh_material='main', indent=1))
    @(tag_link(link_name='Lrl4_foot',   mesh_path='mrl4_foot.stl',  mesh_material='rubber_foot', indent=1))
    <!-- rear right leg -->
    @(tag_link(link_name='Lrr1_hipr',   mesh_path='mrr1_hipr.stl',  mesh_material='main', indent=1))
    @(tag_link(link_name='Lrr2_hipp',   mesh_path='mrr2_hipp.stl',  mesh_material='main', indent=1))
    @(tag_link(link_name='Lrr3_knee',   mesh_path='mrr3_knee.stl',  mesh_material='main', indent=1))
    @(tag_link(link_name='Lrr4_foot',   mesh_path='mrr4_foot.stl',  mesh_material='rubber_foot', indent=1))

    <!-- joint tags  -->
@[if not _no_virtual_links]@
    @(tag_joint(joint_name='J_F_base_L0_torso', indent=1))
    @(tag_joint(joint_name='J_L0_torso_F_imu', indent=1))
@[end if]@
    <!-- front left leg -->
    @(tag_joint(joint_name='Jfl1_hipr', indent=1))
    @(tag_joint(joint_name='Jfl2_hipp', indent=1))
    @(tag_joint(joint_name='Jfl3_knee', indent=1))
    @(tag_joint(joint_name='Jfl4_foot', indent=1))
    <!-- front right leg -->
    @(tag_joint(joint_name='Jfr1_hipr', indent=1))
    @(tag_joint(joint_name='Jfr2_hipp', indent=1))
    @(tag_joint(joint_name='Jfr3_knee', indent=1))
    @(tag_joint(joint_name='Jfr4_foot', indent=1))
    <!-- rear left leg -->
    @(tag_joint(joint_name='Jrl1_hipr', indent=1))
    @(tag_joint(joint_name='Jrl2_hipp', indent=1))
    @(tag_joint(joint_name='Jrl3_knee', indent=1))
    @(tag_joint(joint_name='Jrl4_foot', indent=1))
    <!-- rear right leg -->
    @(tag_joint(joint_name='Jrr1_hipr', indent=1))
    @(tag_joint(joint_name='Jrr2_hipp', indent=1))
    @(tag_joint(joint_name='Jrr3_knee', indent=1))
    @(tag_joint(joint_name='Jrr4_foot', indent=1))

</robot>
