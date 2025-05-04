
import copy
import os
import torch
import numpy as np
import argparse

import sys
sys.path.append("./")
sys.path.append("./datasets")
sys.path.append("./models")

from datasets.dataset import Voxelizer, PolarDataset, collate_fn_BEV
from datasets.dataset_nuscenes import NuscenesForeground

import configs.nuscenes_config as config

from models.vqvae_transformers import VQVAETrans, MaskGIT, voxels2points
import open3d
import pickle
import logging
import timeit

############ app
import gradio as gr
import plotly.graph_objects as go
from demo_app.demo_utils import *

import pandas as pd



#################################################################################################
######################### INTERACTIVE DEMONSTRATION ##############################################
##############################################################################################

REF_FIG_INIT =  "initial LiDAR scene"
REF_FIG_PREV = "previous LiDAR scene"
DEFAULT_CAMERA = dict(
    eye=dict(x=1.25, y=1.25, z=1.25),  # Camera viewpoint
    up=dict(x=0, y=0, z=1),            # Camera up direction
    center=dict(x=0, y=0, z=0)         # Camera focus point
)
DEFAULT_DATASET_SAMPLE_IDX = 17
 ### website demo mini
# k = 17
# k = 33
# k = 130 #137
# k = 290 ### train mini
#######


class DemoApp:
    def __init__(self, allocentric_dict, full_obj_pc_path, train_dataset: PolarDataset, val_dataset: PolarDataset, mask_git: MaskGIT, voxelizer: Voxelizer, device, mode="spherical"):
        self.logger = create_logger("./demo_app", "demo_debug.log")

        ### default dataset choice
        self.dataset_list = {"train":train_dataset, "val":val_dataset}
        self.dataset = train_dataset

        self.mask_git = mask_git
        self.voxelizer = voxelizer
        self.mode = mode
        self.obj_library = allocentric_dict
        self.full_obj_pc_path = full_obj_pc_path
        self.device = device

        ############## Initialize APP ##################
        init_point_cloud, init_bbox_list, ground_points, other_background_points, ground_points_pcd = self.init_from_dataset(DEFAULT_DATASET_SAMPLE_IDX)
        self.logger.info(f"num initial objects: {len(init_bbox_list)}")
        

        #### these are with the initial lidar point cloud for drivable surface rendering
        self.ground_points = ground_points
        self.other_background_points = other_background_points
        self.ground_points_pcd = ground_points_pcd

        ###### APP state ######
        self.state = {"pointcloud": init_point_cloud, "bbox_list":init_bbox_list}

        pcd_colors=np.array([[1,1,1]])
        pcd_colors = np.repeat(pcd_colors, len(init_point_cloud), axis=0)
        init_state_pcd = pcd_ize(init_point_cloud, colors=pcd_colors)
        self.update_geoms(init_state_pcd, bbox2lineset(self.state["bbox_list"])) ### geometries for edited lidar scene visualization
        self.selected_bbox_indices = set() ### indices of object boxes to be removed

        ####### reference figure #####
        self.reference_fig_choice = REF_FIG_INIT
        #### for reference initial lidar scene
        self.initial_geoms = copy.deepcopy(self.geoms) + [self.ground_points_pcd]
        ##### for reference previous lidar scene
        self.previous_geoms = copy.deepcopy(self.geoms)

        ############################################################################

        self.figure_width = 800
        self.figure_height = 800
        self.column_names = ['x', 'y', 'class', 'allocentric angle (degree)', 'insertion Status']

       



    def update_state(self, pointcloud, bbox_list):
        self.state["pointcloud"] = pointcloud
        self.state["bbox_list"] = bbox_list

    def update_geoms(self, pcd, lineset_list):
        assert(isinstance(pcd, open3d.geometry.PointCloud))
        self.geoms = [pcd] + lineset_list
        return self.geoms
    
    def init_from_dataset(self, k: int):
        print(f"NOTE: =============== currently at Sample {k} =========================")
        self.logger.info(f"NOTE: =============== currently at Sample {k} =========================")
        return_from_data = self.dataset.__getitem__(k)
        if return_from_data is None:
            print(f"sample_idx: {k}")
            assert(1==0)

        data_tuple = collate_fn_BEV([return_from_data])
        has, no, voxel_label, BEV_label = data_tuple
        grid_ind_has, return_points_has, voxel_centers_has, voxels_occupancy_has = has
        grid_ind_no, return_points_no, voxel_centers_no, voxels_occupancy_no = no
        voxels_mask = voxel_label.to(device) #(B, H, W, in_chans)
        BEV_mask = BEV_label.to(device) #(B,H,W)

        ####### what we want
        dataset_obj_boxes_list = self.dataset.obj_properties[5] #5
        voxels_occupancy_has = voxels_occupancy_has.permute(0,3,1,2).to(device).float() #(B, in_chans, H, W)
        scene_points_xyz = voxels2points(self.voxelizer, voxels_occupancy_has, mode=self.mode)[0]

        ###### for drivable surface visualization
        ground_points = dataset.point_cloud_dataset.ground_points[:,:3]
        other_background_points = dataset.point_cloud_dataset.other_background_points[:,:3]
        ground_points_colors = np.zeros((len(ground_points), 3))
        ground_points_colors[:,0] = 1 ### red drivable surface points
        ground_points_pcd = pcd_ize(ground_points, colors=ground_points_colors)


        return scene_points_xyz, dataset_obj_boxes_list, ground_points, other_background_points, ground_points_pcd
    
    def reinit_and_render_from_dataset(self, k):
        init_point_cloud, init_bbox_list, ground_points, other_background_points, ground_points_pcd = self.init_from_dataset(k)
        self.logger.info(f"num initial objects: {len(init_bbox_list)}")

        #### these are with the initial lidar point cloud for drivable surface rendering
        self.ground_points = ground_points
        self.other_background_points = other_background_points
        self.ground_points_pcd = ground_points_pcd

        ###### APP state ######
        self.state = {"pointcloud": init_point_cloud, "bbox_list":init_bbox_list}

        pcd_colors=np.array([[1,1,1]])
        pcd_colors = np.repeat(pcd_colors, len(init_point_cloud), axis=0)
        init_state_pcd = pcd_ize(init_point_cloud, colors=pcd_colors)
        self.update_geoms(init_state_pcd, bbox2lineset(self.state["bbox_list"])) ### geometries for edited lidar scene visualization
        self.selected_bbox_indices = set() ### indices of object boxes to be removed

        ####### reference figure #####
        # self.reference_fig_choice = REF_FIG_INIT
        #### for reference initial lidar scene
        self.initial_geoms = copy.deepcopy(self.geoms) + [self.ground_points_pcd]
        ##### for reference previous lidar scene
        self.previous_geoms = copy.deepcopy(self.geoms)

        return self.update_plots(camera_config=DEFAULT_CAMERA)
    
    

    #################### UPDATE GUI: REMOVA: ########################
    
    def update_plots(self, camera_config=None):
        '''
        render edited plot and reference plot
        '''
        edited_fig =  self.make_edited_figure(camera_config)
        ref_fig = self.make_reference_figure_driver(camera_config)
        
        return edited_fig, ref_fig


    def update_box_selection_plots(self, selected_idxs, camera_config=None):
        '''
        heighlight selected boxes
        '''
        self.selected_bbox_indices = set(map(int, selected_idxs)) if selected_idxs else set()
        return self.update_plots(camera_config)

    def update_reference_plot(self, choice):
        '''
        render reference figure according to the string choice
        '''
        self.reference_fig_choice = choice
        ref_fig = self.make_reference_figure_driver()
        return ref_fig

    
    def update_object_removal(self):
        '''
        Remove selected objects indicated by self.selected_bbox_indices
        '''
        if len(self.selected_bbox_indices)==0:
            new_choices = [str(i) for i in range(len(self.state["bbox_list"]))]
            edited_fig, ref_fig = self.update_box_selection_plots([])
            return (
                gr.CheckboxGroup(choices=new_choices, value=[]),
                edited_fig,
                ref_fig,
                "no objects selected"
            )

        to_remove = set(self.selected_bbox_indices)
        old_bbox_list = self.state["bbox_list"]
        new_bbox_list = [old_bbox_list[i] for i in range(len(old_bbox_list)) if i not in to_remove]
        selected_boxes =  [old_bbox_list[i] for i in range(len(old_bbox_list)) if i in to_remove]
        
        scene_points_xyz = self.state["pointcloud"]
        result = inpainting_driver(self.mask_git, scene_points_xyz, selected_boxes, self.voxelizer, device=self.device, mode=self.mode)

        if result is None:
            #### no regions are masked (no occluded regions)
            edited_fig, ref_fig = self.update_box_selection_plots([])
            return (
                gr.CheckboxGroup(choices=[str(i) for i in range(len(self.state["bbox_list"]))], value=[]),
                edited_fig,
                ref_fig,
                 "no occluded region"
            )


        ############# successful object removal
        gen_voxels, gen_points_xyz, gen_pcd = result
        ##### for reference previous lidar scene
        self.previous_geoms = copy.deepcopy(self.geoms)

        #### update state and visuzliation geometries
        self.update_state(gen_points_xyz, new_bbox_list)
        self.update_geoms(gen_pcd, bbox2lineset(new_bbox_list))

        new_choices = [str(i) for i in range(len(self.state["bbox_list"]))]
        edited_fig, ref_fig = self.update_box_selection_plots([], camera_config=DEFAULT_CAMERA) # clear selections, default camera angke
  
        return (
            gr.CheckboxGroup(choices=new_choices, value=[]),
            edited_fig,
            ref_fig,
            ""
        )
    
    def update_remove_all(self):
        self.selected_bbox_indices = {i for i in range(len(self.state["bbox_list"]))}
        return self.update_object_removal()
    
    ##################### UPDATE GUI: INSERTION ################

    def not_inserting(self, table_data):
        new_choices = [str(i) for i in range(len(self.state["bbox_list"]))]
        edited_fig, ref_fig = self.update_plots() ## not changing anything
        return (
            gr.CheckboxGroup(choices=new_choices, value=[]),
            edited_fig,
            ref_fig,
            gr.Dataframe(value=table_data),
            ""
        )
    

    def update_object_insertion(self, table_data):
        '''
        Object insertion according to the dataframe table_data
        '''
        try:
            if not isinstance(table_data, list):
                table_data_list = table_data.values.tolist()
            else:
                table_data_list = table_data

            if len(table_data_list)==0:
                return self.not_inserting(table_data)
            
            self.logger.critical(f"to list: {table_data_list}")

            # Extract columns via list comprehensions:
            xs      = [float(row[0]) for row in table_data_list]
            ys      = [float(row[1]) for row in table_data_list]
            classes = [row[2] for row in table_data_list]
            angles    = [float(row[3]) for row in table_data_list]
            valid_classes = {"car", "bus", "truck"}
            for idx, name in enumerate(classes):
                if (name not in valid_classes):
                    raise Exception(f"{idx}th row: inserted object's class {name} not in {valid_classes}")
            
            ## prepare inputs
            insert_names = classes
            insert_xy_pos_list = [np.array([xs[i], ys[i]]) for i in range(len(xs))]
            insert_alpha_list = [np.deg2rad(angles[i]) for i in range(len(xs))]
            curr_bbox_list = self.state["bbox_list"]

            self.logger.info(f"insert xy pos list: {insert_xy_pos_list}")
            self.logger.info(f"insert alpha list: {insert_alpha_list}")
            self.logger.info(f"insert name list: {insert_names}")
            
            ## voxelize current point cloud
            scene_points_polar = cart2polar(self.state["pointcloud"][:,:3], mode=mode)
            _, _, _, voxels_occupancy = self.voxelizer.voxelize(scene_points_polar, return_point_info=False) #(H, W, in_chans)
            voxels_occupancy_has = torch.tensor(voxels_occupancy).unsqueeze(0).to(self.device).float() #(1, H, W, in_chans)

            ### insert objects
            new_scene_points_xyz, new_bbox_list, old_bbox_list, voxels_occupancy_has, \
            new_points_xyz_no_resampling_occlusion, failure_message_list, \
            failure_indicator_list = insertion_vehicles_driver(insert_names, insert_xy_pos_list, insert_alpha_list, voxels_occupancy_has, curr_bbox_list, self.voxelizer, self.obj_library, self.full_obj_pc_path, self.ground_points, self.other_background_points, mode=self.mode, logger=self.logger)

            # Optional: new_scene_points_xyz = new_points_xyz_no_resampling_occlusion

             ##### for reference previous lidar scene
            self.previous_geoms = copy.deepcopy(self.geoms)
            
            #### update point cloud and bounding box list
            self.state["pointcloud"] = new_scene_points_xyz
            self.state["bbox_list"] = new_bbox_list + old_bbox_list
            pcd = pcd_ize_inserted_pointcloud(new_scene_points_xyz, new_bbox_list)
            self.update_geoms(pcd, bbox2lineset(self.state["bbox_list"]))

            ######### give insertion status 
            assert(len(failure_message_list)==len(table_data_list))
            for i in range(len(table_data_list)):
                table_data_list[i][-1] = failure_message_list[i]
            new_data_with_status = table_data_list
            new_table_data = pd.DataFrame(new_data_with_status, columns=self.column_names)

            new_choices = [str(i) for i in range(len(self.state["bbox_list"]))]
            edited_fig, ref_fig = self.update_box_selection_plots([], camera_config=DEFAULT_CAMERA) ### clear the box selections, default camera angle

            return (
                gr.CheckboxGroup(choices=new_choices, value=[]),
                edited_fig,
                ref_fig,
                gr.Dataframe(value=new_table_data),
                ""
            )
        
        except Exception as e:
            error_message = f"Error: {str(e)}"
            if not isinstance(table_data, list):
                table_data_list = table_data.values.tolist()
            else:
                table_data_list = table_data

            ### show erroneous status in insertion table
            for i in range(len(table_data_list)):
                table_data_list[i][-1] = error_message
            new_data_with_status = table_data_list
            new_table_data = pd.DataFrame(new_data_with_status, columns=self.column_names)

            new_choices = [str(i) for i in range(len(self.state["bbox_list"]))]
            
            self.logger.critical(error_message)
            edited_fig, ref_fig = self.update_box_selection_plots([]) ### clear the box selections
            return (
                gr.CheckboxGroup(choices=new_choices, value=[]),
                edited_fig,
                ref_fig,
                gr.Dataframe(value=new_table_data),
                error_message
            )
    
    def clear_rows(self):
        # just reset table to empty
        new_table_data = pd.DataFrame([["" for _ in range(len(self.column_names))]], columns=self.column_names)
        return gr.Dataframe(value=new_table_data)  # Keep column names intact, clear rows
    

    def update_dataset_sample_menu(self, split):
        '''
        update the dataset sample index drop down menu, depending on which dataset split the user chooses
        '''
        new_choices = [i for i in range(len(self.dataset_list[split]))]
        assert(len(new_choices)!=0)
        return gr.Dropdown(choices= new_choices, value=new_choices[0], interactive=True)
    
    def change_lidar_scene(self, split, sample_idx):
        '''
        change_lidar_scene_button.click(fn=self.change_lidar_scene, inputs=[dataset_split_menu, dataset_sample_menu], outputs=[bbox_selector, edited_plot, ref_plot])
        '''
        # print("split: ", split)
        # print("sample_idx: ", sample_idx)
        assert(split in ["train", "val"])
        self.dataset = self.dataset_list[split]
        edited_fig, ref_fig = self.reinit_and_render_from_dataset(sample_idx)

        new_choices = [str(i) for i in range(len(self.state["bbox_list"]))]
        new_checkbox = gr.CheckboxGroup(
                        choices=new_choices,
                        label="Select Bounding Boxes",
                        value=[]
                    )

        return new_checkbox, edited_fig, ref_fig

    #######################################################

    ################## RENDER FIGURES ###################
    def make_edited_figure(self, camera_config=None):
        return self.make_figure_for_geoms(self.geoms, self.selected_bbox_indices, camera_config)

    def make_figure_for_geoms(self, geoms, selected_bbox_indices, camera_config=None):
        '''
        3D figure for visualizing point cloud and bounding boxes (line set)
        '''
        fig = go.Figure()

        for idx, geom in enumerate(geoms):
            if isinstance(geom, open3d.geometry.PointCloud):
                pts = np.asarray(geom.points)
                cols = np.asarray(geom.colors)
                # self.logger.critical(cols)
                fig.add_trace(go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    mode='markers',
                    marker=dict(
                        size=1,
                        color=[f'rgb({r:.0f},{g:.0f},{b:.0f})' for r, g, b in (cols * 255)],
                    ),
                    hovertemplate="x: %{x:.2f}<br>" +"y: %{y:.2f}<extra></extra>",
                    name="PointCloud"
                ))

            elif isinstance(geom,  open3d.geometry.LineSet):
                pts = np.asarray(geom.points)
                lines = np.asarray(geom.lines)

                # Extract uniform color from the first entry
                if len(geom.colors) > 0:
                    r, g, b = (np.asarray(geom.colors[0]) * 255).astype(int)
                else:
                    r, g, b = 255, 255, 255

                x_lines, y_lines, z_lines = [], [], []
                for i, j in lines:
                    x_lines += [pts[i, 0], pts[j, 0], None]
                    y_lines += [pts[i, 1], pts[j, 1], None]
                    z_lines += [pts[i, 2], pts[j, 2], None]

                bbox_idx = idx - 1 #### point cloud is always index 0
                line_width = 3
                if bbox_idx in selected_bbox_indices:
                    # line_width = 15  # thicker to highlight
                    r, g, b = 0, 255, 0

                fig.add_trace(go.Scatter3d(
                    x=x_lines, y=y_lines, z=z_lines,
                    mode='lines',
                    line=dict(width=line_width, color=f'rgb({r},{g},{b})'),
                    name=f"Box_{bbox_idx}"
                ))
            else:
                raise Exception("what .....")
            
        if camera_config is not None:
            scene_dict = dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                aspectmode='data',
                xaxis=dict(
                    showbackground=True,
                    backgroundcolor='black',
                    gridcolor='gray',      # optional: grid lines
                    zerolinecolor='gray',   # optional: zero line
                    title=dict(font=dict(color='white')),  # Title "X" color
                    tickfont=dict(color='white')           # Numbers color
                ),
                # Y-axis plane (XZ) background
                yaxis=dict(
                    showbackground=True,
                    backgroundcolor='black',
                    gridcolor='gray',
                    zerolinecolor='gray',
                    title=dict(font=dict(color='white')),  # Title "X" color
                    tickfont=dict(color='white')           # Numbers color
                ),
                # Z-axis plane (XY) background (the “floor”)
                zaxis=dict(
                    showbackground=True,
                    backgroundcolor='black',
                    gridcolor='gray',
                    zerolinecolor='gray',
                    title=dict(font=dict(color='white')),  # Title "X" color
                    tickfont=dict(color='white')           # Numbers color
                ),
                camera=camera_config
            )
        else:
            scene_dict = dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                aspectmode='data',
                xaxis=dict(
                    showbackground=True,
                    backgroundcolor='black',
                    gridcolor='gray',      # optional: grid lines
                    zerolinecolor='gray',   # optional: zero line
                    title=dict(font=dict(color='white')),  # Title "X" color
                    tickfont=dict(color='white')           # Numbers color
                ),
                # Y-axis plane (XZ) background
                yaxis=dict(
                    showbackground=True,
                    backgroundcolor='black',
                    gridcolor='gray',
                    zerolinecolor='gray',
                    title=dict(font=dict(color='white')),  # Title "X" color
                    tickfont=dict(color='white')           # Numbers color
                ),
                # Z-axis plane (XY) background (the “floor”)
                zaxis=dict(
                    showbackground=True,
                    backgroundcolor='black',
                    gridcolor='gray',
                    zerolinecolor='gray',
                    title=dict(font=dict(color='white')),  # Title "X" color
                    tickfont=dict(color='white')           # Numbers color
                )
            )

        fig.update_layout(
            scene=scene_dict,
            paper_bgcolor='black',  # ← whole figure background
            #plot_bgcolor='black',    # ← 3D box background
            font=dict(color='white'),

            
            width=self.figure_width,    # ← total pixel width of the figure
            height=self.figure_height,    # ← total pixel height of the figure
            margin=dict(l=0, r=0, t=30, b=0),
            dragmode='orbit',
            hovermode='closest',
            title="PointCloud + Bounding Boxes Viewer",
            uirevision='KEEP_CAMERA',
            showlegend=False
 
        )

        return fig
    
    def make_initial_reference_figure(self, camera_config=None):
        '''
        render initial lidar scene
        '''
        static_pc_fig = self.make_figure_for_geoms(self.initial_geoms, [], camera_config)
        return static_pc_fig
    
    def make_previous_reference_figure(self, camera_config=None):
        '''
        render previous lidar scene
        '''
        prev_pc_fig = self.make_figure_for_geoms(self.previous_geoms, [], camera_config)
        return prev_pc_fig
    
    def make_reference_figure_driver(self, camera_config=None):
        '''
        render reference plot
        '''
        if self.reference_fig_choice==REF_FIG_INIT:
            ref_fig = self.make_initial_reference_figure(camera_config)
        elif self.reference_fig_choice==REF_FIG_PREV:
            ref_fig = self.make_previous_reference_figure(camera_config)
        else:
            raise gr.Error(f"Invalid reference figure name: {self.reference_fig_choice}")
        
        return ref_fig
    
    ##############################################################################

    def launch(self):
        # render both figrues first
        edited_fig, ref_fig = self.update_plots()

        with gr.Blocks() as demo:
            #gr.Markdown("# LiDAR-EDIT: Lidar Data Generation by Editing the Object Layouts in Real-World Scenes")
            gr.Markdown("""
            <div style='text-align: center;'>
                <h1>LiDAR-EDIT: Lidar Data Generation by Editing the Object Layouts in Real-World Scenes</h1>
            </div>
            """)
            gr.Markdown("""
            # Instructions
            This is an interactive demo of the paper.
            ## Rendering LiDAR scene
                - use the dropdowns to select a dataset split and sample to choose / reset the initial LiDAR scene.
                - Choose which reference scene to visualize from the reference figure choice dropdown menu.
                - The most recent edited LiDAR scene is shown on the right.
            ## Color scheme:
                - Red: default color of bounding boxes and points on drivable surface
                - Green: color of selected bounding boxes, edited points
            ## Object removal:
                - Select bounding boxes of objects to be removed, and then click one of the object removal buttons
            ## Object insertion:
                - Hover over the plots to view the x-y coordinates of points in the scene. Insertion that collides with background points or other bounding bxoes would be rejected. The drivable surface's point cloud can help you decide where to insert objects. 
                - Fill in the x-y coordinates, class (car, bus or truck) and the allocentric angle of the object to be inserted, and then click the object insertion button. You can insert multiple objects at once
                - Whether each object insertion is successful would be shown in the status column of the table
                - errors will be shown in the error message text box

            ### What is allocentric angle? 
                - In Bird-eye view, given a vector R pointing from a LiDAR sensor to the object center. Given another Vector A pointing from the object center to the right of the object. Allocentric angle is the angle between the vector R and vector A.  
            """)


            assert(DEFAULT_DATASET_SAMPLE_IDX in [i for i in range(len(self.dataset))])
            with gr.Row():
                dataset_split_menu = gr.Dropdown(label="Dataset Split", choices=["train", "val"], value="train", interactive=True)
                dataset_sample_menu = gr.Dropdown(label="Dataset Sample Index", choices=[i for i in range(len(self.dataset))], value=DEFAULT_DATASET_SAMPLE_IDX, interactive=True)
                change_lidar_scene_button = gr.Button("Change initial Lidar scene")

            ref_view_selection_dropdown = gr.Dropdown(choices=[REF_FIG_INIT, REF_FIG_PREV], value=REF_FIG_INIT, label="reference figure choice")

             # --- Row 1: Plots ---
            with gr.Row():
                with gr.Column(scale=2):
                    ref_plot = gr.Plot(label="Reference LiDAR Scene", value=ref_fig)
                with gr.Column(scale=2):
                    edited_plot = gr.Plot(label="Edited LiDAR Scene", value=edited_fig)

            error_textbox = gr.Textbox(label="error message", value="")

            # --- Row 2: Insertion Controls ---
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Object Insertion Controls")
                    insertion_table = gr.Dataframe(
                        headers=self.column_names,
                        datatype=["number", "number", "str", "number", "str"],
                        row_count=(1, "dynamic"),
                        interactive=True,
                        label="Objects to Insert"
                    )
                    with gr.Row():
                        clear_rows_btn = gr.Button("🗑️ Clear All Rows")
                        insert_button   = gr.Button("Insert Object(s)")

            # --- Row 3: Removal Controls ---
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Object Removal Controls")
                    bbox_selector = gr.CheckboxGroup(
                        choices=[str(i) for i in range(len(self.state["bbox_list"]))],
                        label="Select Bounding Boxes",
                        value=[]
                    )
                    with gr.Row():
                        remove_button      = gr.Button("Remove Selected Object(s)")
                        remove_all_button  = gr.Button("Remove All Object(s)")

            ########## Events #####################
            dataset_split_menu.change(fn=self.update_dataset_sample_menu, inputs=dataset_split_menu, outputs=dataset_sample_menu)
            ref_view_selection_dropdown.change(self.update_reference_plot, inputs=ref_view_selection_dropdown, outputs=ref_plot)

            change_lidar_scene_button.click(fn=self.change_lidar_scene, inputs=[dataset_split_menu, dataset_sample_menu], outputs=[bbox_selector, edited_plot, ref_plot])

            bbox_selector.change(
                fn=self.update_box_selection_plots,
                inputs=bbox_selector,
                outputs=[edited_plot, ref_plot]
            )

            remove_button.click(
                fn=self.update_object_removal,
                inputs=None,
                outputs=[bbox_selector, edited_plot, ref_plot, error_textbox]
            )

            remove_all_button.click(
                fn=self.update_remove_all,
                inputs=None,
                outputs=[bbox_selector, edited_plot, ref_plot, error_textbox]
            )

            clear_rows_btn.click(self.clear_rows,  inputs=None, outputs=insertion_table)
            insert_button.click(self.update_object_insertion, inputs=insertion_table, outputs=[bbox_selector, edited_plot, ref_plot, insertion_table, error_textbox])
            ###########################################

        demo.launch(share=True)









if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--trainval_data_path', default="./data_mini", type=str, help="path to training and validation examples")
    parser.add_argument('--data_version', default="v1.0-mini", type=str, help="versioin of the dataset e.g. nuscenes")
    parser.add_argument('--maskgit_path', default="./weights/maskgit_trans_weights/epoch_44", type=str, help="path to the trained maskGIT's weights of a specific epoch")
    parser.add_argument('--vqvae_path', default="./weights/vqvae_trans_weights/epoch_60", type=str, help="path to the trained vqvae's weights of a specific epoch")
    parser.add_argument('--full_obj_pc_path', default="./foreground_object_pointclouds", type=str, help="path to full vehicle point clouds")
    args = parser.parse_args()



    config.device = "cpu"
    device = torch.device(config.device)
    print("--- device: ", device)

    ######### use spherical or polar coordinates
    mode = config.mode
    use_z = False
    if mode=="spherical":
        use_z=True
    assert(mode=="spherical")

    ############ initialize voxelizer and NuScenes dataset
    vis = False
    voxelizer = Voxelizer(grid_size=config.grid_size, max_bound=config.max_bound, min_bound=config.min_bound)
    ## important: if you are just inserting the original object point cloud to debug, set filter_obj_point_with_seg to False
    train_pt_dataset = NuscenesForeground(args.trainval_data_path, version = args.data_version, split = 'train', vis=vis, mode=mode, voxelizer=voxelizer, filter_obj_point_with_seg=False, get_raw=True, any_scene=True)
    val_pt_dataset = NuscenesForeground(args.trainval_data_path, version = args.data_version, split = 'val', vis =vis, mode=mode, voxelizer=voxelizer, filter_obj_point_with_seg=False, get_raw=True, any_scene=True)
    train_dataset=PolarDataset(train_pt_dataset, voxelizer, flip_aug = False, rotate_aug = False, is_test=True, vis=vis, insert=True)
    val_dataset=PolarDataset(val_pt_dataset, voxelizer, flip_aug = False, rotate_aug = False, is_test=True, vis=vis, insert=True)


    ############ load trained models ###########
    vqvae_config = config.vqvae_trans_config
    window_size=vqvae_config["window_size"]
    patch_size=vqvae_config["patch_size"]
    patch_embed_dim = vqvae_config["patch_embed_dim"]
    num_heads = vqvae_config["num_heads"]
    depth = vqvae_config["depth"]
    codebook_dim = vqvae_config["codebook_dim"]
    num_code = vqvae_config["num_code"]
    beta = vqvae_config["beta"]
    dead_limit = vqvae_config["dead_limit"]
    vqvae = VQVAETrans(
        img_size=voxelizer.grid_size[0:2],
        in_chans=voxelizer.grid_size[2],
        patch_size=patch_size,
        window_size=window_size,
        patch_embed_dim=patch_embed_dim,
        num_heads=num_heads,
        depth=depth,
        codebook_dim=codebook_dim,
        num_code=num_code,
        beta=beta,
        device=device,
        dead_limit=dead_limit
    ).to(device)
    vqvae.load_state_dict(torch.load(args.vqvae_path)["model_state_dict"])

    maskgit_config = config.maskgit_trans_config
    mask_git = MaskGIT(vqvae=vqvae, voxelizer=voxelizer, hidden_dim=maskgit_config["hidden_dim"], depth=maskgit_config["depth"], num_heads=maskgit_config["num_heads"]).to(device)
    mask_git.load_state_dict(torch.load(args.maskgit_path)["model_state_dict"])
    
    #### get blank codes for blank code suppressing 
    gen_blank_code = True
    if gen_blank_code:
        print("generating blank code")
        mask_git.get_blank_code(path=".", name="blank_code", iter=100)

    print(f"--- num blank code: {len(mask_git.blank_code)}")
    print(f"--- blank code: {mask_git.blank_code}")

    
    dataset = train_dataset
    samples = np.arange(len(dataset))

    torch.cuda.empty_cache()
    seed = 100 #300 for website demo #100
    torch.manual_seed(seed)
    np.random.seed(seed)

    ################# load object library
    
    with open(os.path.join(args.full_obj_pc_path, "allocentric.pickle"), 'rb') as handle:
        allocentric_dict = pickle.load(handle)

    for name in allocentric_dict.keys():
        allocentric_dict[name][0] = np.array(allocentric_dict[name][0])
        allocentric_dict[name][2] = np.array(allocentric_dict[name][2])
        allocentric_dict[name][4] = np.array(allocentric_dict[name][4])
        allocentric_dict[name][12] = np.array(allocentric_dict[name][12])

    
    
   
    demo = DemoApp(allocentric_dict, args.full_obj_pc_path, train_dataset, val_dataset, mask_git, voxelizer, device, mode)
    demo.launch()

    # table_data = [[-2.03, -16.88, "car", 0]]
    # demo.update_object_insertion(table_data)