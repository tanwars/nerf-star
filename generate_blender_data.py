import argparse, sys, os
import json
import bpy
import mathutils
import numpy as np
import math

from math import radians

DEBUG = False
            
#VIEWS = 200
#RESOLUTION = 800
VIEWS = 1
RESOLUTION = 1
KEYFRAMES = 2
SKIP = 1 # skip k keyframes for loop range(start, start+skip*keyframes, SKIP)
RESULTS_PATH = 'data_dynamic_testing'
DEPTH_SCALE = 1.4
COLOR_DEPTH = 8
FORMAT = 'PNG'
UPPER_VIEWS = True
CIRCLE_FIXED_START = (0,0,0)
CIRCLE_FIXED_END = (.7,0,0)

fp = bpy.path.abspath(f"//{RESULTS_PATH}")

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

def get_keyframes(obj_list):
    keyframes = []
    for obj in obj_list:
        anim = obj.animation_data
        if anim is not None and anim.action is not None:
            for fcu in anim.action.fcurves:
                for keyframe in fcu.keyframe_points:
                    x, y = keyframe.co
                    if x not in keyframes:
                        keyframes.append((math.ceil(x)))
    return keyframes

def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty
    # scn.objects.active = b_empty
    return b_empty

#def parent_obj_to_ball(b_ball, scn):
#    origin = (0, 0, 0)
#    b_empty = bpy.data.objects.new("Empty", None)
#    b_empty.location = origin
#    b_ball.parent = b_empty  # setup parenting

#    # scn = bpy.context.scene
#    scn.collection.objects.link(b_empty)
#    bpy.context.view_layer.objects.active = b_empty
#    # scn.objects.active = b_empty
#    return b_empty

### main

if not os.path.exists(fp):
    os.makedirs(fp)

# Render Optimizations
bpy.context.scene.render.use_persistent_data = True

# Background
bpy.context.scene.render.dither_intensity = 0.0
bpy.context.scene.render.film_transparent = True

# Create collection for objects not to render with background
objs = [ob for ob in bpy.context.scene.objects if ob.type in ('EMPTY') and 'Empty' in ob.name]
bpy.ops.object.delete({"selected_objects": objs})

scene = bpy.context.scene
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100

stepsize = 360.0 / VIEWS
vertical_diff = CIRCLE_FIXED_END[0] - CIRCLE_FIXED_START[0]
rotation_mode = 'XYZ'

cam = scene.objects['Camera']
cam.location = (0, 4.0, 0.5)
cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
b_empty = parent_obj_to_camera(cam)
cam_constraint.target = b_empty

scene.render.image_settings.file_format = 'PNG'

# Data to store in JSON file
out_data = {
    'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
}
out_data['keyframes'] = []

out_data_old_format = {
    'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
}
out_data_old_format['frames'] = []

# Background
bpy.context.scene.render.dither_intensity = 0.0
bpy.context.scene.render.film_transparent = True
bpy.context.scene.render.image_settings.color_mode = 'RGBA'

for keyframe_idx in range(scene.frame_start,scene.frame_start + SKIP * KEYFRAMES, SKIP):
    print('*'*80)
    print(f'Keyframe: {keyframe_idx}')
    print('*'*80)
    
    scene.frame_current = keyframe_idx
    
    b_empty.rotation_euler = CIRCLE_FIXED_START
    b_empty.rotation_euler[0] = CIRCLE_FIXED_START[0] + vertical_diff

    frames_views = []
    for i in range(0, VIEWS):
        print(f'     On iteration: {i}')
        print("Rotation {}, {}".format((stepsize * i), radians(stepsize * i)))
        scene.render.filepath = fp + '/r_' + str(i) + '_t_' + str(keyframe_idx)
        
        bpy.ops.render.render(write_still=True)
        
        frame_data = {
            'file_path': './train' + '/r_' + str(i) + '_t_' + str(keyframe_idx),
            'rotation': radians(stepsize),
            'transform_matrix': listify_matrix(cam.matrix_world)
        }
        frames_views.append(frame_data)
        out_data_old_format['frames'].append(frame_data)
        
        b_empty.rotation_euler[0] = CIRCLE_FIXED_START[0] + (np.cos(radians(stepsize*i))+1)/2 * vertical_diff
        b_empty.rotation_euler[2] += radians(2*stepsize)
    
    ball = scene.objects['SurfSphere']
#    print(ball.matrix_world.decompose())
    keyframe_data = {
        'frames': frames_views,
        't_mat_dynamic': listify_matrix(ball.matrix_world)
    }
    out_data['keyframes'].append(keyframe_data)

with open(fp + '/' + 'transforms.json', 'w') as out_file:
    json.dump(out_data, out_file, indent=4)

with open(fp + '/' + 'transforms_old_format.json', 'w') as out_file:
    json.dump(out_data_old_format, out_file, indent=4)

## get all selected objects
#selection = bpy.context.scene.objects['SurfSphere']

## check if selection is not empty
#if selection:

#    # get all frames with assigned keyframes
#    keys = get_keyframes([selection])

#    # print all keyframes
#    print (keys)   

#    # print first and last keyframe
#    print ("{} {}".format("first keyframe:", keys[0]))
#    print ("{} {}".format("last keyframe:", keys[-1]))

#else:
#    print ('nothing selected')

