#!/usr/bin/env python3

###############################################################################
# Copyright 2018 The Apollo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################



"""
dns add path
"""
import sys
import os

# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


"""
dns add path end 
"""
import argparse

import matplotlib.pyplot as plt

# from modules.tools.mapshow.libs.localization import Localization
# from modules.tools.mapshow.libs.map import Map
# from modules.tools.mapshow.libs.path import Path


from modules.tools.mapshow.libs.localization import Localization
from modules.tools.mapshow.libs.map import Map
from modules.tools.mapshow.libs.path import Path

def draw(map):
    lane_ids = args.laneid
    if lane_ids is None:
        lane_ids = []
    map.draw_lanes(plt, args.showlaneids, lane_ids, args.showlanedetails)
    # if args.showsignals:
    #     map.draw_signal_lights(plt)
    # if args.showstopsigns:
    #     map.draw_stop_signs(plt)
    # if args.showjunctions:
    #     map.draw_pnc_junctions(plt)
    # if args.showcrosswalks:
    #     map.draw_crosswalks(plt)
    # if args.showyieldsigns:
    #     map.draw_yield_signs(plt)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Mapshow is a tool to display hdmap info on a map.",
        prog="mapshow.py")

    parser.add_argument(
        "-m", "--map", action="store", type=str, required=False, default = "modules/map/data/yizhuang/base_map.bin",
        help="Specify the map file in txt or binary format")
    parser.add_argument(
        "-m2", "--map2", action="store", type=str, required=False,
        help="Specify the map file in txt or binary format")
    parser.add_argument(
        "-sl", "--showlaneids", action="store_const", const=True,
        help="Show all lane ids in map")
    parser.add_argument(
        "-sld", "--showlanedetails", action="store_const", const=True,
        help="Show all lane ids in map")
    parser.add_argument(
        "-l", "--laneid", nargs='+',
        help="Show specific lane id(s) in map")
    parser.add_argument(
        "-signal", "--showsignals", action="store_const", const=True,
        help="Show all signal light stop lines with ids in map")
    parser.add_argument(
        "-stopsign", "--showstopsigns", action="store_const", const=True,
        help="Show all stop sign stop lines with ids in map")
    parser.add_argument(
        "-yieldsign", "--showyieldsigns", action="store_const", const=True,
        help="Show all yield sign stop lines with ids in map")
    parser.add_argument(
        "-junction", "--showjunctions", action="store_const", const=True,
        help="Show all pnc-junctions with ids in map")
    parser.add_argument(
        "-crosswalk", "--showcrosswalks", action="store_const", const=True,
        help="Show all crosswalks with ids in map")
    parser.add_argument(
        "--loc", action="store", type=str, required=False,
        help="Specify the localization pb file in txt format")
    parser.add_argument(
        "--position", action="store", type=str, required=False,
        help="Plot the x,y coordination in string format, e.g., 343.02,332.01")

    # driving path data files are text files with data format of
    # t,x,y,heading,speed
    parser.add_argument(
        "-dp", "--drivingpath", nargs='+',
        help="Show driving paths in map")

    args = parser.parse_args()
    """dns """
    # plt.figure(figsize=(9 ,12))
    fig, ax =plt.subplots()
    print("begin")
    map = Map()
    map.load(args.map)
    draw(map)



    # plt.figure(figsize=(4, 4))

    if args.map2 is not None:
        map2 = Map()
        print("Loading map")
        map2.load(args.map2)
        print("Loading map")
        draw(map2)

    if args.drivingpath is not None:
        path = Path(args.drivingpath)
        path.draw(plt)

    if args.loc is not None:
        localization = Localization()
        localization.load(args.loc)
        localization.plot_vehicle(plt)

    if args.position is not None:
        x, y = args.position.split(",")
        x, y = float(x), float(y)
        plt.plot([x], [y], 'bo')
    
    # all map 2
    fig.set_size_inches(12,18)
    plt.xlim(458250, 458650)
    plt.ylim(4399900,4400500)

    plt.rcParams["lines.color"]='black'
    plt.savefig("./dnstest/yizhuangall2",transparent =False)
    plt.savefig("./dnstest/yizhuangall2_t",transparent =True)
    print("save1yizhuangall2")



    fig.set_size_inches(5,15)
    plt.xlim(458500, 458600)
    plt.ylim(4399900,4400200)
    # plt.axis('equal')
    # plt.figure(figsize=(4, 4))
    plt.rcParams["lines.color"]='black'
    plt.savefig("./dnstest/case1_map",transparent =False)
    plt.savefig("./dnstest/case1_map_t",transparent =True)
    print("save2case1_map")

    fig.set_size_inches(12,8)
    plt.xlim(458350, 458500)
    plt.ylim(4400200,4400300)
    # plt.axis('equal')
    # plt.figure(figsize=(4, 4))
    plt.rcParams["lines.color"]='black'
    plt.savefig("./dnstest/case2_map",transparent =False)
    plt.savefig("./dnstest/case2_map_t",transparent =True)
    print("save2case2_map")


    fig.set_size_inches(4,10)
    plt.xlim(458250, 458450)
    plt.ylim(4400000,4400500)
    # plt.axis('equal')
    # plt.figure(figsize=(4, 4))
    plt.rcParams["lines.color"]='black'
    plt.savefig("./dnstest/case3_map",transparent =False)
    plt.savefig("./dnstest/case3_map_t",transparent =True)
    print("save2case3_map")
    # plt.show()
