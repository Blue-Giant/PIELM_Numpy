import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from utilizers import DNN_tools


def plot_solu_1D(solu=None, coords=None, color='m', actName='diff', outPath=None):
    fig11 = plt.figure(figsize=(9, 6.5))
    ax = plt.gca()
    ax.plot(coords, solu, color=color, linestyle='dotted', label=actName)
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    ax.legend(loc='right', bbox_to_anchor=(0.9, 1.05), ncol=4, fontsize=12)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('u', fontsize=14)
    fntmp = '%s/solus2%s' % (outPath, actName)
    DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)


def plot_2solus_1D(exact_solu=None, nn_solu=None,  coords=None, batch_size=1000, outPath=None, subfig_type=1):
    if subfig_type == 1:
        plt.figure(figsize=(16, 10), dpi=98)
        fig, ax = plt.subplots(1, 1)  # fig, ax = plt.subplots(a,b)用来控制子图个数：a为行数，b为列数。
        ax.plot(coords, exact_solu, 'b-.', label='exact')
        ax.plot(coords, nn_solu, 'g:', label='nn')
        ax.legend(fontsize=10)
        ax.set_xlabel('x', fontsize=18)

        axins = inset_axes(ax, width="50%", height="40%", loc=8, bbox_to_anchor=(0.2, 0.4, 0.5, 0.5),
                           bbox_transform=ax.transAxes)

        # 在子坐标系中绘制原始数据
        axins.plot(coords, exact_solu, color='b', linestyle='-.')

        axins.plot(coords, nn_solu, color='g', linestyle=':')

        axins.set_xticks([])
        axins.set_yticks([])

        # 设置放大区间
        zone_left = int(0.4 * batch_size)
        zone_right = int(0.4 * batch_size) + 100

        # 坐标轴的扩展比例（根据实际数据调整）
        x_ratio = 0.0  # x轴显示范围的扩展比例
        y_ratio = 0.075  # y轴显示范围的扩展比例

        # X轴的显示范围
        xlim0 = coords[zone_left] - (coords[zone_right] - coords[zone_left]) * x_ratio
        xlim1 = coords[zone_right] + (coords[zone_right] - coords[zone_left]) * x_ratio

        # Y轴的显示范围
        y = np.hstack((exact_solu[zone_left:zone_right], nn_solu[zone_left:zone_right]))
        ylim0 = np.min(y) - (np.max(y) - np.min(y)) * y_ratio
        ylim1 = np.max(y) + (np.max(y) - np.min(y)) * y_ratio

        # 调整子坐标系的显示范围
        axins.set_xlim(xlim0, xlim1)
        axins.set_ylim(ylim0, ylim1)

        # 建立父坐标系与子坐标系的连接线
        # loc1 loc2: 坐标系的四个角
        # 1 (右上) 2 (左上) 3(左下) 4(右下)
        mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)

        fntmp = '%s/solus2test' % (outPath)
        DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)
    elif subfig_type == 2:
        plt.figure(figsize=(16, 10), dpi=98)
        ax = plt.gca()
        p1 = plt.subplot(121)  # 1行2列，第一个图
        p2 = plt.subplot(122)  # 1行2列，第二个图

        p1.plot(coords, exact_solu, color='b', linestyle='-.', label='exact')
        p1.plot(coords, nn_solu, color='g', linestyle=':', label='nn')
        ax.legend(fontsize=10)

        p2.plot(coords, exact_solu, color='b', linestyle='-.', label='exact')
        p2.plot(coords, nn_solu, color='g', linestyle=':', label='nn')
        p2.axis([0.35, 0.65, 0.2, 0.27])

        # plot the box of
        tx0 = 0.35
        tx1 = 0.65
        ty0 = 0.2
        ty1 = 0.27
        sx = [tx0, tx1, tx1, tx0, tx0]
        sy = [ty0, ty0, ty1, ty1, ty0]
        p1.plot(sx, sy, "purple")

        # plot patch lines
        xy = (0.64, 0.265)
        xy2 = (0.36, 0.265)
        con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data", axesA=p2, axesB=p1)
        p2.add_artist(con)

        xy = (0.64, 0.21)
        xy2 = (0.36, 0.205)
        con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                              axesA=p2, axesB=p1)
        p2.add_artist(con)

        fntmp = '%s/solus2test' % (outPath)
        DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)
    else:
        fig11 = plt.figure(figsize=(9, 6.5))
        ax = plt.gca()
        ax.plot(coords, exact_solu, 'b-.', label='exact')
        ax.plot(coords, nn_solu, 'r:', label='nn')
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
        ax.legend(loc='right', bbox_to_anchor=(0.9, 1.05), ncol=4, fontsize=12)
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('u', fontsize=14)
        fntmp = '%s/solus2test' % (outPath)
        DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)


def plot_scatter_solu(solu, test_batch, color='b', actName=None, outPath=None):
    test_x_bach = np.reshape(test_batch[:, 0], newshape=[-1, 1])
    test_y_bach = np.reshape(test_batch[:, 1], newshape=[-1, 1])

    # 绘制解的3D散点图
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = Axes3D(fig)
    ax.scatter(test_x_bach, test_y_bach, solu, c=color, label=actName)

    # 刻度值字体大小设置（x轴和y轴同时设置）
    plt.tick_params(labelsize=18)
    # 绘制图例
    ax.legend(loc='best', fontsize=16)
    # 添加坐标轴(顺序是X，Y, Z)
    ax.set_xlabel('X', fontdict={'size': 16, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 16, 'color': 'red'})
    ax.set_zlabel('u', fontdict={'size': 16, 'color': 'red'})

    # plt.title('solution', fontsize=15)
    fntmp = '%s/solus2%s' % (outPath, actName)
    DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)


def plot_scatter_solus(solu1, solu2, test_batch, actName1=None, actName2=None, outPath=None):
    test_x_bach = np.reshape(test_batch[:, 0], newshape=[-1, 1])
    test_y_bach = np.reshape(test_batch[:, 1], newshape=[-1, 1])

    # 绘制解的3D散点图(真解和预测解)
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = Axes3D(fig)
    ax.scatter(test_x_bach, test_y_bach, solu1, c='b', label=actName1)
    ax.scatter(test_x_bach, test_y_bach, solu2, c='r', label=actName2)

    # 刻度值字体大小设置（x轴和y轴同时设置）
    plt.tick_params(labelsize=18)
    # 绘制图例
    ax.legend(loc='best', fontsize=16)
    # 添加坐标轴(顺序是X，Y, Z)
    ax.set_xlabel('X', fontdict={'size': 16, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 16, 'color': 'red'})
    ax.set_zlabel('u', fontdict={'size': 16, 'color': 'red'})

    # plt.title('solution', fontsize=15)
    fntmp = '%s/solus2%s' % (outPath, actName2)
    DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)
