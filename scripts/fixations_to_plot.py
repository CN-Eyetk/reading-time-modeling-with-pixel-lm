import numpy as np
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from matplotlib.text import TextPath
from PIL import Image, ImageFilter, ImageDraw
from matplotlib.path import Path
import matplotlib as mpl
import skimage.io
import skimage.filters
from skimage import filters
import skimage.draw
from matplotlib.backends.backend_agg import FigureCanvasAgg
from skimage import color

def get_char_widths(text, fontproperties, font_size, ax):
    widths = []

    fig = ax.figure
    fig.canvas.draw()  # 必须先 draw 一次，bbox 才是正确的
    renderer = fig.canvas.get_renderer()

    for char in text:
        # 先画一个临时文本
        text_obj = ax.text(0, 0, char, fontproperties=fontproperties, fontsize=font_size)

        # 获取字符的 bbox（像素）
        bbox = text_obj.get_window_extent(renderer=renderer)
        char_width_pixel = bbox.width
        # print("char_width_pixel", char_width_pixel)

        # --- 像素宽度转成 data 宽度 ---
        # 选择一个参考点（随便，只要在轴内）：
        p0_disp = ax.transData.transform((0, 0))  # data(0,0) -> display 坐标
        # 在 display 坐标中往右偏移 char_width_pixel 像素
        p1_disp = p0_disp + np.array([char_width_pixel, 0])

        # 再把两个点都反变换回 data 坐标
        p0_data = ax.transData.inverted().transform(p0_disp)
        p1_data = ax.transData.inverted().transform(p1_disp)

        # 宽度 = 两个 data 点的 x 差
        char_width_data = p1_data[0] - p0_data[0]
        widths.append(char_width_data)

        text_obj.remove()

    return widths

def adjust_position_2ds_by_char(ias, position_2ds):
    ia_with_position_2ds = list(zip(ias, position_2ds))
    sorted_ia_with_position_2ds = sorted(ia_with_position_2ds, key = lambda x: (x[1][0], x[1][1]))
    ias = [sorted_ia_with_position_2d[0] for sorted_ia_with_position_2d in sorted_ia_with_position_2ds]
    position_2ds = [sorted_ia_with_position_2d[1] for sorted_ia_with_position_2d in sorted_ia_with_position_2ds]
    adjusted_position_2ds = []
    cur_x = 0
    last_y = -1
    for ia, position_2d in zip(ias, position_2ds):
        cur_row = position_2d[0]
        
        if cur_row > last_y:
            cur_x = 0
        cur_col = cur_x
        last_y = cur_row
        adjusted_position_2ds.append((cur_row, cur_col))
        cur_x += len(ia)
    return ias, adjusted_position_2ds

def get_visual_length(fp, text):
    # Create a TextPath object
    text_path = TextPath((0, 0), text, prop=fp)
    # Get the bounding box
    bbox = text_path.get_extents() 
    # Extract width and height
    width = bbox.width
    height = bbox.height
    return width, height
def text_to_plot(
                ias,
                wns,#must start from 0
                lps,#Must be proportionate
                durs,
                char_width,
                space_width,
                margin,
                y_base,
                position_2ds,
                ia_process_func = None,
                position_by_char = True,
                ia_only = False,
                adjust_to_visual_length = False,
                fp = None,
                visual_width_function = None,
                width_limit_by_axis = None
                ):
    if adjust_to_visual_length:
        assert visual_width_function is not None
    labels = []
    orths = []
    xs = []
    x_rights = []
    ys = []
    path_data = []
    cs = []
    #boxes = []
    #print("len ia",len(ias))
    #print("len position_2ds",len(position_2ds))

    if position_by_char:
        ias, position_2ds = adjust_position_2ds_by_char(ias, position_2ds)
    cur_x = 0
    last_y = None

    for i, (ia, position_2d) in enumerate(zip(ias, position_2ds)):
        if ia_process_func is None:
            pure_ia = ia#remove repeat label
        else:
            pure_ia = ia_process_func(ia)
        row, col = position_2d
        row = 5 - row
        y = (row + 1) * char_width + (row) * margin + y_base
        if last_y and y < last_y:
            x = 0
        else:
            x = cur_x
        
        if adjust_to_visual_length:
            #fig, ax = plt.subplots(figsize=(10, 6))
            
            if space_width > 0:
                x_right = x + visual_width_function(pure_ia) + visual_width_function(" ")#sum(get_char_widths(" ", fontproperties = fontproperties, ax = ax)).item()
            else:
                x_right = x + visual_width_function(pure_ia)
        else:
            #if not adjust_to_visual_length:
            # + len(pure_ia) * char_width + space_width#col * char_width + i * space_width
            x_right = x + len(pure_ia) * char_width + space_width

        xs.append(x)
        
        #x_lps.append(min(x + lps[i], x + len(pure_ia) * char_width))
        
        x_rights.append(x_right)
        cur_x = x_right
        
        ys.append(y)
        
        labels.append(ia)
        orths.append(pure_ia)
        last_y = y
        if ia_only:
            cs.append("black")
        elif i in wns:
            if wns.count(i) > 1:
                cs.append("brown")
            else:
                cs.append("black")
        else:
            cs.append("grey")
        #boxes.append((x, y - 0.5 * char_width))
    if width_limit_by_axis is not None:
        max_x_right = max(x_rights)
        if max_x_right > width_limit_by_axis:
            print(f"Raw width is {max_x_right}, but limit is {width_limit_by_axis}. Now using offset")
            offset = max_x_right - width_limit_by_axis
            xs = [x - offset for x in xs]
            x_rights = [x_right - offset for x_right in x_rights]
    ia_data = list(zip(labels, orths, xs, x_rights, ys, cs))
    if ia_only:
        return ia_data
    #print("labels",labels)
    last_x_y = None
    for i in range(len(wns)-1):
        start_wn = wns[i]
        end_wn = wns[i+1]
        start_ia = ias[wns[i]]
        end_ia = ias[wns[i+1]]
        #print("start_ia",start_ia)
        #print("end_ia",end_ia)
        start_order = start_wn
        end_order = end_wn
        start_ia_width =  x_rights[start_order] - xs[start_order]
        end_ia_width = x_rights[end_order] - xs[end_order]
        lp_start = min(xs[start_order] + lps[i] * start_ia_width, x_rights[start_order])#x_lps[start_order]
        lp_end = min(xs[end_order] + lps[i+1] * end_ia_width, x_rights[end_order])#x_lps[end_order]
        if last_x_y is None:
            start_x_y = lp_start, ys[start_order]
        else:
            start_x_y = last_x_y
        end_x_y = lp_end, ys[end_order] + np.random.normal(0, char_width/15)
        #if last_x_y is None:
        last_x_y = end_x_y
        start_dur = durs[i]
        end_dur = durs[i+1]
        path_data.append((start_x_y, end_x_y, start_ia, end_ia, start_dur, end_dur))
    
    return ia_data, path_data

def plot_on_ax(ax, ia_data, path_data, title, fontproperties, font_size, xlim, ylim, draw_box = True):
    if len(xlim) == 2:
        ax.set_xlim(xlim)#1200 if lang == "tg" else 1600)
    else:
        ax.set_xlim(0, xlim)#1200 if lang == "tg" else 1600)
    if len(ylim) == 2:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(50, ylim)
    
    ax.set_title(title, fontproperties=fontproperties)
    #
    for label, orth, x, x_right, y, c in ia_data:
        if draw_box:
            rect = patches.Rectangle((x, y - 0.5 * font_size), x_right - x, font_size, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        ax.text(x = x, y = y, s = orth, c=c,       
            horizontalalignment='left',
            fontsize=font_size,#FONT_SIZE[lang],
            verticalalignment='center',fontproperties=fontproperties)
        
    n_fix = 0
    if path_data is not None:
        for i,path in enumerate(path_data):
            start = path[0]
            end = path[1]
            start_dur = path[4]
            end_dur = path[5]
            if start == end:
                pass
            else:
                ax.plot((start[0],end[0]), (start[1],end[1]), '--k',mfc='C1', mec='C1',linewidth=0.5)
                #ax.arrow(np.mean([start[0], end[0]]), np.mean([start[1],end[1]]),diff_x/np.abs(diff_x), slope*diff_x/np.abs(diff_x), head_width=2, head_length = 5, shape='full', lw=1, length_includes_head=True)
                if i == 0:
                    ax.scatter(x=start[0], y=start[1], color="blue", s = start_dur/5, alpha = min(end_dur / 500, 0.5))
                    ax.text(x=start[0], y=start[1], color="black", s = n_fix, fontsize = 10,         
                            horizontalalignment='center',
                            verticalalignment='center',)
                    n_fix += 1
                ax.scatter(x=end[0], y=end[1], color="blue", s = end_dur/5, alpha = min(end_dur / 500, 0.5))
                ax.text(x=end[0], y=end[1], color="black", s = n_fix, fontsize = 10,horizontalalignment='center',verticalalignment='center',)
                n_fix += 1
    ax.axis('off')


def gaze_plot(ia_data, path_data, title, output_path, fontproperties, font_size, xlim, ylim, figsize = (10,7), draw_box = True):
    #prop = fm.FontProperties()
    #prop.set_family("Songti SC")

    fig, ax = plt.subplots(figsize = figsize)
    plot_on_ax(ax, ia_data, path_data, title, fontproperties, font_size, xlim, ylim, draw_box = draw_box)
    #plt.show()
    if output_path is not None:
        plt.savefig(os.path.join(output_path, f"{title}.png"),dpi=200)
    else:
        plt.show()


class GazePlotter:
    def __init__(self, 
                char_width = 56, 
                space_width = 0,
                margin = 5,
                y_base = 0,
                xlim = 1600,
                ylim = 500,
                font_size = 24,
                font = "Songti SC",
                output_path = None,
                figsize = [10,7],
                adjust_to_visual_length = True,
                use_dataset_native_parameters = False
                ) -> None:
        self.char_width = char_width
        self.space_width = space_width
        self.margin = margin
        self.y_base = y_base
        self.xlim = xlim
        self.ylim = ylim
        self.font_size = font_size
        self.prop = fm.FontProperties(fname=font)
        #self.prop.set_family(font)
        self.output_path = output_path
        self.figsize = figsize
        self.adjust_to_visual_length = adjust_to_visual_length
        self.use_dataset_native_parameters = use_dataset_native_parameters

    def get_image(self,return_array = False):
        return mpl_to_skimage(self.fig, return_array = return_array)
    def draw_fixation_report(self, ias, 
                                wns, 
                                lps, 
                                durs, 
                                position_2ds, 
                                title= "", 
                                draw_box = True, 
                                mask = None, 
                                crop = False,
                                crop_margin = 5
                                ):
        if self.adjust_to_visual_length:
            _, ax = plt.subplots(figsize=(self.figsize[0], self.figsize[1]))
            ax.set_xlim(0, self.xlim)#1200 if lang == "tg" else 1600)
            ax.set_ylim(50, self.ylim)
            visual_width_function = lambda x:sum(get_char_widths(x, self.prop, self.font_size, ax = ax)).item()
        else:
            visual_width_function = None

        ia_data, path_data = text_to_plot(ias = ias, wns = wns, lps = lps, durs = durs, 
                                        char_width = self.char_width, 
                                        space_width = self.space_width, 
                                        margin = self.margin, 
                                        y_base = self.y_base, 
                                        position_2ds = position_2ds, 
                                        adjust_to_visual_length = self.adjust_to_visual_length,
                                        visual_width_function = visual_width_function
                                        )
        if self.adjust_to_visual_length:
            ax.remove()
        #gaze_plot(ia_data, 
        #        path_data,
        #        title=title,
        #        output_path=self.output_path,
        #        fontproperties=self.prop,
        #        font_size=self.font_size,
        #        xlim = self.xlim,
        #        ylim = self.ylim,
        #        figsize=(self.figsize[0], self.figsize[1])
        fig, ax = plt.subplots(figsize=(self.figsize[0], self.figsize[1]))

        if crop:
            min_x = min([x[2] for x in ia_data]) - crop_margin
            #max_x = max([x[3] for x in ia_data]) + crop_margin
            min_y = min([x[4] for x in ia_data]) - crop_margin
            max_y = max([x[4] for x in ia_data]) + crop_margin
            xlim = (min_x, self.xlim)
            ylim = (min_y, max_y)
        else:
            xlim = (0, self.xlim)
            ylim = (0, self.ylim)
            
        plot_on_ax(
            ax, ia_data, path_data=path_data, title=title,
            fontproperties=self.prop, font_size=self.font_size,
            xlim=xlim, ylim=ylim, draw_box=draw_box
        )

        # ✅ 新增：mask blur（mask==0 的 IA 区域会被高斯模糊）
        #self._apply_text_fade(ax, ia_data, mask=mask, blur_radius_px=blur_radius_px)
        #self.ax = ax
        #self.fig = fig
        self.ia_data = ia_data
        self.path_data = path_data
        
        if self.output_path is not None:
            plt.savefig(os.path.join(self.output_path, f"{title}.png"), dpi=200)
        else:
            plt.show()

        return ia_data, path_data
    def draw_ia_plot(self, ias, 
                    position_2ds, 
                    title,
                    draw_box = True,
                    mask=None,
                    gradient=None,
                    mask_effect="fade",
                    show = True,
                    crop = False,
                    crop_margin = 5,
                    width_limit_by_axis = None
                    ):
        fig, ax = plt.subplots(figsize=(self.figsize[0], self.figsize[1]))
        if self.adjust_to_visual_length:
            #_, ax = plt.subplots(figsize=(self.figsize[0], self.figsize[1]))
            ax.set_xlim(0, self.xlim)#1200 if lang == "tg" else 1600)
            ax.set_ylim(50, self.ylim)
            visual_width_function = lambda x:sum(get_char_widths(x, self.prop, self.font_size, ax = ax)).item()
        else:
            visual_width_function = None
        
        ia_data = text_to_plot(ias = ias, wns = None, lps = None, durs = None, 
                                        char_width = self.char_width, 
                                        space_width = self.space_width, 
                                        margin = self.margin, 
                                        y_base = self.y_base, 
                                        position_2ds = position_2ds, 
                                        ia_only=True, 
                                        adjust_to_visual_length = self.adjust_to_visual_length,
                                        visual_width_function = visual_width_function,
                                        width_limit_by_axis = width_limit_by_axis

                                        )
        
        if crop:
            min_x = min([x[2] for x in ia_data]) - crop_margin
            #max_x = max([x[3] for x in ia_data]) + crop_margin
            min_y = min([x[4] for x in ia_data]) - crop_margin
            max_y = max([x[4] for x in ia_data]) + crop_margin
            xlim = (min_x, self.xlim)
            ylim = (min_y, max_y)
        else:
            xlim = (0, self.xlim)
            ylim = (0, self.ylim)
 
        plot_on_ax(
            ax, ia_data, path_data=None, title=title,
            fontproperties=self.prop, font_size=self.font_size,
            xlim=xlim, ylim=ylim, draw_box=draw_box
        )
        
        
        self.ax = ax
        self.fig = fig
        self.ia_data = ia_data

        
        if self.output_path is not None:
            plt.savefig(os.path.join(self.output_path, f"{title}.png"), dpi=200)
        elif show:
            plt.show()

        return ia_data

    def _overlay_saliency(
        self, ax, ia_data, saliency,
        cmap="viridis", vmin=None, vmax=None,
        alpha=0.30, add_colorbar=False,
        colorbar_label="saliency",
        # ===== 新增：渲染风格 =====
        style="color",  # "color" | "alpha" | "fog" | "fog_texture"
        # ===== alpha / fog 参数 =====
        alpha_mode="constant",          # "constant" | "by_saliency_transparent"（仅 style="alpha" 时用）
        alpha_min=0.05, alpha_max=0.60, # alpha_mode 时：saliency高->alpha_min(更透明)
        facecolor_when_alpha="black",   # style="alpha" 时的填充颜色（只靠透明度表达）
        # ===== fog 参数（雾：用白色半透明层“蒙住”）=====
        fog_color="white",
        fog_min=0.05,   # saliency高 -> 雾薄（更透明）
        fog_max=0.75,   # saliency低 -> 雾厚（更不透明）
        # ===== fog_texture 参数（雾+轻微纹理，更像雾气）=====
        fog_noise_strength=0.12,  # 0.05~0.15 比较自然
        fog_seed=0,
        # ===== 层级 =====
        zorder=80,
    ):

        sal = np.asarray(saliency, dtype=float)
        if len(sal) != len(ia_data):
            raise ValueError(f"saliency length {len(sal)} != ia_data length {len(ia_data)}")

        finite = np.isfinite(sal)
        if vmin is None:
            vmin = np.nanmin(sal) if finite.any() else 0.0
        if vmax is None:
            vmax = np.nanmax(sal) if finite.any() else 1.0
        if vmin == vmax:
            vmax = vmin + 1e-9

        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cm = mpl.cm.get_cmap(cmap)

        def s01(x):
            return float(np.clip(norm(x), 0.0, 1.0))

        def alpha_for(s):
            if alpha_mode == "constant":
                return float(alpha)
            if alpha_mode == "by_saliency_transparent":
                # saliency高 -> 更透明 -> alpha更小
                t = s01(s)
                return float(alpha_max + (alpha_min - alpha_max) * t)
            raise ValueError(f"Unknown alpha_mode: {alpha_mode}")

        def fog_alpha_for(s):
            # saliency高 -> 雾薄；saliency低 -> 雾厚
            t = s01(s)
            return float(fog_max + (fog_min - fog_max) * t)

        # -------- style dispatch --------
        if style not in ("color", "alpha", "fog", "fog_texture"):
            raise ValueError(f"Unknown style: {style}")

        # colorbar：只有 color 风格才默认有意义；alpha/fog 更像视觉遮罩，色条会误导
        want_cbar = add_colorbar and (style == "color")

        if style in ("color", "alpha", "fog"):
            for i, (label, orth, x, x_right, y, c) in enumerate(ia_data):
                if not np.isfinite(sal[i]):
                    continue

                if style == "color":
                    fc = cm(norm(sal[i]))
                    a = float(alpha)

                elif style == "alpha":
                    fc = facecolor_when_alpha
                    a = alpha_for(sal[i])

                else:  # style == "fog"
                    fc = fog_color
                    a = fog_alpha_for(sal[i])

                rect = patches.Rectangle(
                    (x, y - 0.5 * self.font_size),
                    (x_right - x),
                    self.font_size,
                    linewidth=0,
                    edgecolor="none",
                    facecolor=fc,
                    alpha=float(np.clip(a, 0.0, 1.0)),
                    zorder=zorder
                )
                ax.add_patch(rect)

        else:
            # style == "fog_texture": 每个词框叠一张白雾纹理（灰度图+alpha）
            rng = np.random.default_rng(fog_seed)
            for i, (label, orth, x, x_right, y, c) in enumerate(ia_data):
                if not np.isfinite(sal[i]):
                    continue

                a = fog_alpha_for(sal[i])

                # 生成局部噪声（尺寸跟词框相关）
                w = max(10, int((x_right - x) / 2))
                h = max(10, int(self.font_size / 2))
                noise = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
                noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)  # 0..1

                # 变成接近 1 的“白雾纹理”
                # fog_noise_strength 越大，纹理越明显
                fog = 1.0 - fog_noise_strength + fog_noise_strength * noise  # ~[1-s,1]

                ax.imshow(
                    fog,
                    extent=(x, x_right, y - 0.5 * self.font_size, y + 0.5 * self.font_size),
                    origin="lower",
                    cmap="gray",
                    vmin=0, vmax=1,
                    alpha=float(np.clip(a, 0.0, 1.0)),
                    interpolation="bilinear",
                    zorder=zorder,
                    aspect="auto"
                )

        if want_cbar:
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
            sm.set_array([])
            cbar = ax.figure.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label(colorbar_label)

    def draw_fixation_plot_with_saliency(self, ias, wns, lps, durs, position_2ds,saliency, title,
                                path_data = None,
                                cmap="viridis", vmin=None, vmax=None,
                                alpha=0.30, add_colorbar=False,
                                colorbar_label="saliency",
                                draw_box = True,
                                mask=None,
                                blur_radius_px=4,
                                style="color",
                                facecolor_when_alpha="white",
                                alpha_mode="constant"
                                
                                ):
        if self.adjust_to_visual_length:
            fig_tmp, ax_tmp = plt.subplots(figsize=(self.figsize[0], self.figsize[1]))
            ax_tmp.set_xlim(0, self.xlim)
            ax_tmp.set_ylim(50, self.ylim)
            visual_width_function = lambda x: sum(get_char_widths(x, self.prop, self.font_size, ax=ax_tmp)).item()
        else:
            visual_width_function = None
        if path_data is not None:
            ia_data = text_to_plot(
                ias=ias, wns=wns, lps=None, durs=None,
                char_width=self.char_width,
                space_width=self.space_width,
                margin=self.margin,
                y_base=self.y_base,
                position_2ds=position_2ds,
                ia_only=True,
                adjust_to_visual_length=self.adjust_to_visual_length,
                visual_width_function=visual_width_function
            )

        else:
            ia_data, path_data = text_to_plot(ias = ias, wns = wns, lps = lps, durs = durs, 
                                            char_width = self.char_width, 
                                            space_width = self.space_width, 
                                            margin = self.margin, 
                                            y_base = self.y_base, 
                                            position_2ds = position_2ds, 
                                            adjust_to_visual_length = self.adjust_to_visual_length,
                                            visual_width_function = visual_width_function
                                            )
        if self.adjust_to_visual_length:
            plt.close(fig_tmp)
        # 最终绘图：overlay -> 原 plot_on_ax
        fig, ax = plt.subplots(figsize=(self.figsize[0], self.figsize[1]))
        if style == "color":
            self._overlay_saliency(ax, ia_data, saliency,
                                cmap=cmap, vmin=vmin, vmax=vmax,
                                alpha=alpha, add_colorbar=add_colorbar,
                                colorbar_label=colorbar_label, 
                                style=style,
                                facecolor_when_alpha=facecolor_when_alpha,
                                alpha_mode=alpha_mode
                                )

        plot_on_ax(ax, ia_data, path_data=path_data, title=title,
                fontproperties=self.prop, font_size=self.font_size,
                xlim=self.xlim, ylim=self.ylim, draw_box = draw_box)
        if style in ["fog","alpha","fog_texture"]:
            self._overlay_saliency(ax, ia_data, saliency,
                                cmap=cmap, vmin=vmin, vmax=vmax,
                                alpha=alpha, add_colorbar=add_colorbar,
                                colorbar_label=colorbar_label, 
                                style=style,
                                facecolor_when_alpha=facecolor_when_alpha,
                                alpha_mode=alpha_mode
                                )
        if self.output_path is not None:
            plt.savefig(os.path.join(self.output_path, f"{title}.png"), dpi=200)
        else:
            plt.show()

        return ia_data
    def draw_ia_plot_with_saliency(self, ias, position_2ds, saliency, title,
                                cmap="viridis", vmin=None, vmax=None,
                                alpha=0.30, add_colorbar=False,
                                colorbar_label="saliency",
                                draw_box = True,
                                mask=None,
                                blur_radius_px=4,
                                style="color",
                                facecolor_when_alpha="white",
                                alpha_mode="constant"
                                ):
        import matplotlib.pyplot as plt
        import os

        # 复用你原来 draw_ia_plot 的 ia_data 生成逻辑
        if self.adjust_to_visual_length:
            fig_tmp, ax_tmp = plt.subplots(figsize=(self.figsize[0], self.figsize[1]))
            ax_tmp.set_xlim(0, self.xlim)
            ax_tmp.set_ylim(50, self.ylim)
            visual_width_function = lambda x: sum(get_char_widths(x, self.prop, self.font_size, ax=ax_tmp)).item()
        else:
            visual_width_function = None

        ia_data = text_to_plot(
            ias=ias, wns=None, lps=None, durs=None,
            char_width=self.char_width,
            space_width=self.space_width,
            margin=self.margin,
            y_base=self.y_base,
            position_2ds=position_2ds,
            ia_only=True,
            adjust_to_visual_length=self.adjust_to_visual_length,
            visual_width_function=visual_width_function
        )

        if self.adjust_to_visual_length:
            plt.close(fig_tmp)

        # 最终绘图：overlay -> 原 plot_on_ax
        fig, ax = plt.subplots(figsize=(self.figsize[0], self.figsize[1]))

        if style == "color":
            self._overlay_saliency(ax, ia_data, saliency,
                                cmap=cmap, vmin=vmin, vmax=vmax,
                                alpha=alpha, add_colorbar=add_colorbar,
                                colorbar_label=colorbar_label, 
                                style=style,
                                facecolor_when_alpha=facecolor_when_alpha,
                                alpha_mode=alpha_mode
                                )
        plot_on_ax(ax, ia_data, path_data=None, title=title,
                fontproperties=self.prop, font_size=self.font_size,
                xlim=self.xlim, ylim=self.ylim, draw_box = draw_box)
        if style in ["fog","alpha","fog_texture"]:
            self._overlay_saliency(ax, ia_data, saliency,
                                cmap=cmap, vmin=vmin, vmax=vmax,
                                alpha=alpha, add_colorbar=add_colorbar,
                                colorbar_label=colorbar_label, 
                                style=style,
                                facecolor_when_alpha=facecolor_when_alpha,
                                alpha_mode=alpha_mode
                                )
        #self._apply_text_fade(ax, ia_data, mask=mask, blur_radius_px=blur_radius_px)
        if self.output_path is not None:
            plt.savefig(os.path.join(self.output_path, f"{title}.png"), dpi=200)
        else:
            plt.show()

        return ia_data
    def get_boxes(self):
        bboxes = ia_edges_to_skimage_bboxes(self.fig, self.ax, self.ia_data, font_size_pt=self.font_size)
        return bboxes
        
        
    def add_mask_to_ia(self, mask_idx, sigma, gradient = False, 
                    use_sigmoid = False, 
                    power = 1, 
                    k = 8, 
                    custom_function = None, 
                    use_alpha = False, 
                    alpha = 0.5,
                    feather_sigma = 1.5,
                    alpha_max = 1.0,
                    show = False,
                    normalize_t = True,
                    is_rtl = False
                    ):
        bboxes = ia_edges_to_skimage_bboxes(self.fig, self.ax, self.ia_data, font_size_pt=self.font_size)
        if not gradient:
            masked_img = add_mask(self.fig, self.ax, bboxes, mask_idx, sigma, 
                                use_alpha = use_alpha, 
                                alpha = alpha
                                )
        else:
            masked_img = add_mask_gradient(self.fig, 
                                        self.ax, 
                                        bboxes, 
                                        mask_idx, 
                                        sigma, 
                                        use_sigmoid = use_sigmoid,
                                        power = power,
                                        k = k,
                                        custom_function = custom_function,
                                        feather_sigma=feather_sigma,
                                        alpha_max = alpha_max,
                                        normalize_t =normalize_t,
                                        is_rtl = is_rtl
                                        )
        if show:
            fig, ax = plt.subplots(figsize=(self.figsize[0], self.figsize[1]))
            ax.imshow(masked_img)
            ax.axis("off")
        return masked_img, bboxes
        




def overlay_saliency_heatmap(ax, ia_data, saliency, cmap="viridis",
                            vmin=None, vmax=None, alpha=0.6,
                            add_colorbar=True, colorbar_label="saliency",
                            nan_color=(0, 0, 0, 0)):
    """
    在现有 ax 上，按 ia_data 的 (x, x_right, y) 位置画一层热力底色矩形。
    不修改你原来的 plot_on_ax；通常先调用这个，再调用 plot_on_ax 覆盖文字/框/路径。
    """
    sal = np.asarray(saliency, dtype=float)
    if len(sal) != len(ia_data):
        raise ValueError(f"saliency length {len(sal)} != ia_data length {len(ia_data)}")

    finite = np.isfinite(sal)
    if vmin is None:
        vmin = np.nanmin(sal) if finite.any() else 0.0
    if vmax is None:
        vmax = np.nanmax(sal) if finite.any() else 1.0
    if vmin == vmax:
        vmax = vmin + 1e-9

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cm = mpl.cm.get_cmap(cmap)

    def to_rgba(v):
        if not np.isfinite(v):
            return nan_color
        return cm(norm(v))

    # 画底色矩形（不画边框，避免跟你红框冲突）
    for i, (label, orth, x, x_right, y, c) in enumerate(ia_data):
        h = 0.5  # 这里用“数据坐标高度”，更稳：直接取你原来 red box 用的 font_size 会混单位
        # ✅ 更推荐：用 ia_data 自己的 y 与你 plot_on_ax 的 box 高度一致
        # 但你 plot_on_ax 用的是 (y - 0.5*font_size, height=font_size) —— font_size 是点单位
        # 为了不改你原逻辑，这里做一个“近似一致”的底色：用 y 上下各 0.5*font_size 的 data 高度不可靠
        # 所以我们直接用一个略窄的“条带”，不会遮挡文字，视觉上也像 heatmap。
        # 如果你强烈希望完全贴合红框高度，我下面给你“精确贴合版本”（B节）。

        rect = patches.Rectangle(
            (x, y - 0.35),            # 细条带
            (x_right - x),
            0.70,
            linewidth=0,
            edgecolor="none",
            facecolor=to_rgba(sal[i]),
            alpha=alpha,
            zorder=0                  # 放到底层
        )
        ax.add_patch(rect)

    if add_colorbar:
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
        mappable.set_array([])
        cbar = ax.figure.colorbar(mappable, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label(colorbar_label)

    return ax


def overlay_saliency_heatmap_match_box(ax, ia_data, saliency, font_size,
                                       cmap="viridis", vmin=None, vmax=None,
                                       alpha=0.35, add_colorbar=True,
                                       colorbar_label="saliency",
                                       nan_color=(0, 0, 0, 0)):
    import numpy as np
    import matplotlib as mpl
    import matplotlib.patches as patches

    sal = np.asarray(saliency, dtype=float)
    if len(sal) != len(ia_data):
        raise ValueError(f"saliency length {len(sal)} != ia_data length {len(ia_data)}")

    finite = np.isfinite(sal)
    if vmin is None:
        vmin = np.nanmin(sal) if finite.any() else 0.0
    if vmax is None:
        vmax = np.nanmax(sal) if finite.any() else 1.0
    if vmin == vmax:
        vmax = vmin + 1e-9

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cm = mpl.cm.get_cmap(cmap)

    def to_rgba(v):
        if not np.isfinite(v):
            return nan_color
        return cm(norm(v))

    # 完全按你 plot_on_ax 的红框参数来
    for i, (label, orth, x, x_right, y, c) in enumerate(ia_data):
        rect = patches.Rectangle(
            (x, y - 0.5 * font_size),
            (x_right - x),
            font_size,
            linewidth=0,
            edgecolor="none",
            facecolor=to_rgba(sal[i]),
            alpha=alpha,
            zorder=0
        )
        ax.add_patch(rect)

    if add_colorbar:
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
        mappable.set_array([])
        cbar = ax.figure.colorbar(mappable, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label(colorbar_label)

    return ax

def gaze_plot_with_saliency(ia_data, path_data, saliency, title, output_path,
                            fontproperties, font_size, xlim, ylim,
                            figsize=(10,7),
                            cmap="viridis", vmin=None, vmax=None,
                            alpha=0.35, add_colorbar=True,
                            match_box=True):
    fig, ax = plt.subplots(figsize=figsize)

    # 先画底色热力
    if match_box:
        overlay_saliency_heatmap_match_box(
            ax, ia_data, saliency, font_size=font_size,
            cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha,
            add_colorbar=add_colorbar
        )
    else:
        overlay_saliency_heatmap(
            ax, ia_data, saliency,
            cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha,
            add_colorbar=add_colorbar
        )

    # 再用你原来的函数画框/字/scanpath（覆盖在上面）
    plot_on_ax(ax, ia_data, path_data, title, fontproperties, font_size, xlim, ylim)

    if output_path is not None:
        plt.savefig(os.path.join(output_path, f"{title}.png"), dpi=200)
    else:
        plt.show()


def mpl_to_skimage(fig, return_array = False):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    ncols, nrows = canvas.get_width_height()
    img_array = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 4)
    img_rgb = color.rgba2rgb(img_array)
    if return_array:
        return img_array
    else:
        return img_rgb

def ia_edges_to_skimage_bboxes(fig, ax, ia_data, font_size_pt=28, pad_px=2):
    """
    ia_data: [(label, orth, x_left, x_right, y_center, color), ...]  (data coords)
    returns: list of dicts with skimage-ready bbox: (r0,r1,c0,c1)
    """
    fig.canvas.draw()
    W, H = fig.canvas.get_width_height()   # W=ncols, H=nrows (pixels)
    text_h_px = font_size_pt * fig.dpi / 72.0

    bboxes = []
    for label, orth, xL, xR, yC, c in ia_data:
        (xL_px, yC_px) = ax.transData.transform((xL, yC))
        (xR_px, _    ) = ax.transData.transform((xR, yC))

        y0_px = yC_px - 0.5 * text_h_px
        y1_px = yC_px + 0.5 * text_h_px

        c0 = int(np.floor(min(xL_px, xR_px) - pad_px))
        c1 = int(np.ceil (max(xL_px, xR_px) + pad_px))
        r0 = int(np.floor((H - 1 - y1_px) - pad_px))
        r1 = int(np.ceil ((H - 1 - y0_px) + pad_px))

        r0 = max(0, min(H-1, r0))
        r1 = max(0, min(H-1, r1))
        c0 = max(0, min(W-1, c0))
        c1 = max(0, min(W-1, c1))

        bboxes.append({
            "label": label, "orth": orth,
            "r0": r0, "r1": r1, "c0": c0, "c1": c1,
        })
    return bboxes



def add_mask_old(fig, ax, bboxes, mask_idx, sigma, use_alpha = False, alpha = 0.5):
    #bboxes = ia_edges_to_skimage_bboxes(fig, ax, ia_data, font_size_pt=font_size)
    image = mpl_to_skimage(fig)
    image_float = skimage.img_as_float(image)
    mask = np.zeros(image_float.shape[:2], dtype=bool)
    r0s = []
    c0s = []
    r1s = []
    c1s = []
    for idx in mask_idx:
        target_box = bboxes[idx]
        r0 = target_box["r0"]
        c0 = target_box["c0"]
        r1 = target_box["r1"]
        c1 = target_box["c1"]
        r0s.append(r0)
        c0s.append(c0)
        r1s.append(r1)
        c1s.append(c1)
    r0 = min(r0s)
    c0 = min(c0s)
    r1 = max(r1s)
    c1 = max(c1s)
    start = (r0,c0)
    end = (r1,c1)
    rr, cc = skimage.draw.rectangle(start,end, shape=image_float.shape)
    mask[rr, cc] = True
    #sigma = 2.0 # Adjust blur intensity
    if use_alpha:
        img_gray = color.rgb2gray(image_float)
        img_rgb = color.gray2rgb(img_gray)
        alpha_ = np.ones(img_gray.shape) * 255 # 或者 * 1.0
        out = np.dstack((img_rgb, alpha_))
        alpha_channel = out[:, :, 3]
        alpha_channel[r0:r1, c0:c1] = alpha
        out[:, :, 3] = alpha_channel
    else:
        blurred_image = skimage.filters.gaussian(image_float, sigma=(sigma, sigma), channel_axis=-1)
        out = np.where(mask[:, :, None], blurred_image, image_float)
    return out


def add_mask(fig, ax, bboxes, mask_idx, sigma=2.0, use_alpha=False, alpha=0.5):
    """
    对指定 bbox 区域添加 mask。

    参数
    ----
    fig, ax :
        matplotlib 的 figure 和 axis。
        这里保留 ax 是为了和你原来的接口一致，虽然当前函数里没有直接用到 ax。
    bboxes : list[dict]
        每个元素是一个 bbox，格式例如：
        {
            "r0": ...,
            "c0": ...,
            "r1": ...,
            "c1": ...
        }
    mask_idx : list[int]
        需要被 mask 的 bbox 下标列表。
    sigma : float
        高斯模糊强度，仅在 use_alpha=False 时有效。
    use_alpha : bool
        是否使用 alpha 遮罩模式。
    alpha : float
        alpha 值，通常范围是 0~1。
        例如：
        - 0.0 表示完全透明
        - 0.5 表示半透明
        - 1.0 表示完全不透明

    返回
    ----
    out : ndarray
        处理后的图像。
        - use_alpha=False 时，返回 RGB 图像
        - use_alpha=True 时，返回 RGBA 图像
    """

    # 1. 把 matplotlib figure 转成 skimage 可处理的图像
    image = mpl_to_skimage(fig)
    image_float = skimage.img_as_float(image)

    # 2. 建立 2D mask
    h, w = image_float.shape[:2]
    mask = np.zeros((h, w), dtype=bool)

    # 3. 对每个 bbox 单独加到 mask 上
    for idx in mask_idx:
        target_box = bboxes[idx]

        r0 = int(target_box["r0"])
        c0 = int(target_box["c0"])
        r1 = int(target_box["r1"])
        c1 = int(target_box["c1"])

        # 防止越界
        r0 = max(0, min(r0, h - 1))
        c0 = max(0, min(c0, w - 1))
        r1 = max(0, min(r1, h - 1))
        c1 = max(0, min(c1, w - 1))

        # 防止坐标顺序反了
        if r1 < r0:
            r0, r1 = r1, r0
        if c1 < c0:
            c0, c1 = c1, c0

        # 注意 shape 传二维，不要传 image_float.shape
        rr, cc = draw.rectangle(
            start=(r0, c0),
            end=(r1, c1),
            shape=(h, w)
        )
        mask[rr, cc] = True

    # 4. 根据模式生成输出
    if use_alpha:
        # 转灰度，再转回 RGB，做“灰掉 + alpha 遮罩”的效果
        img_gray = color.rgb2gray(image_float)
        img_rgb = color.gray2rgb(img_gray)

        # alpha 通道要和 float 图像一致，范围用 0~1
        alpha_channel = np.ones((h, w), dtype=float)

        # 只在 mask 区域修改 alpha
        alpha_channel[mask] = alpha

        out = np.dstack((img_rgb, alpha_channel))

    else:
        # 整张图先做模糊，再只取 mask 区域替换
        blurred_image = filters.gaussian(
            image_float,
            sigma=(sigma, sigma),
            channel_axis=-1
        )

        out = np.where(mask[:, :, None], blurred_image, image_float)

    return out
def sigmoid(t, k=8):
    return 1 / (1 + np.exp(-k * (t - 0.5)))

def add_mask_gradient_old(fig, ax, bboxes, mask_idx, sigma, power=2, k=8,
                      use_sigmoid=False, custom_function=None,
                      feather_sigma=1.5, alpha_max=0.85):
    image = mpl_to_skimage(fig)
    image = skimage.img_as_float(image)

    blurred = skimage.filters.gaussian(image, sigma=(sigma, sigma), channel_axis=-1)

    H, W = image.shape[:2]
    alpha = np.zeros((H, W), dtype=float)

    #box = bboxes[mask_idx]
    #r0, r1 = int(box["r0"]), int(box["r1"])
    #c0, c1 = int(box["c0"]), int(box["c1"])

    r0s = []
    c0s = []
    r1s = []
    c1s = []
    for idx in mask_idx:
        target_box = bboxes[idx]
        r0 = target_box["r0"]
        c0 = target_box["c0"]
        r1 = target_box["r1"]
        c1 = target_box["c1"]
        r0s.append(r0)
        c0s.append(c0)
        r1s.append(r1)
        c1s.append(c1)
    r0 = min(r0s)
    c0 = min(c0s)
    r1 = max(r1s)
    c1 = max(c1s)
    start = (r0,c0)
    end = (r1,c1)
    for x in range(c0, c1):
        t = (x - c0) / max(c1 - c0, 1)
        if custom_function:
            a = custom_function(t)
        elif use_sigmoid:
            a = sigmoid(t, k=k)
        else:
            a = t ** power

        a = float(np.clip(a, 0.0, alpha_max))   # 限制最大 alpha，减少“脏”
        alpha[r0:r1, x] = a

    # 关键：羽化 bbox 边界，去掉肮脏矩形边
    if feather_sigma and feather_sigma > 0:
        alpha = skimage.filters.gaussian(alpha, sigma=feather_sigma)
        alpha = np.clip(alpha, 0.0, 1.0)

    # 只在 bbox 内混合，bbox 外完全不变
    out = image.copy()
    a_bbox = alpha[r0:r1, c0:c1][..., None]
    out[r0:r1, c0:c1, :] = (1 - a_bbox) * image[r0:r1, c0:c1, :] + a_bbox * blurred[r0:r1, c0:c1, :]
    return out
def add_mask_gradient(fig, ax, bboxes, mask_idx, sigma, power=2, k=8,
                      use_sigmoid=False, custom_function=None,
                      feather_sigma=1.5, alpha_max=0.85, row_tol=8, normalize_t = True, is_rtl = False):
    image = mpl_to_skimage(fig)
    image = skimage.img_as_float(image)

    blurred = skimage.filters.gaussian(image, sigma=(sigma, sigma), channel_axis=-1)

    H, W = image.shape[:2]
    alpha = np.zeros((H, W), dtype=float)

    # ---- 按行分组 ----
    items = []
    for idx in mask_idx:
        box = bboxes[idx]
        cy = 0.5 * (box["r0"] + box["r1"])
        items.append((idx, cy))

    items.sort(key=lambda x: x[1])

    rows = []
    current_row = []
    current_center = None

    for idx, cy in items:
        if current_center is None:
            current_row = [idx]
            current_center = cy
        elif abs(cy - current_center) <= row_tol:
            current_row.append(idx)
            current_center = np.mean([0.5 * (bboxes[i]["r0"] + bboxes[i]["r1"]) for i in current_row])
        else:
            rows.append(current_row)
            current_row = [idx]
            current_center = cy

    if current_row:
        rows.append(current_row)

    # ---- 每一行单独加 alpha_function ----
    for row in rows:
        r0 = max(0, min(int(bboxes[i]["r0"]) for i in row))
        r1 = min(H, max(int(bboxes[i]["r1"]) for i in row))
        c0 = max(0, min(int(bboxes[i]["c0"]) for i in row))
        c1 = min(W, max(int(bboxes[i]["c1"]) for i in row))

        width = max(c1 - c0, 1)
        
        for x in range(c0, c1):
            if is_rtl:
                if normalize_t:
                    t = (c1 - x) / width
                else:
                    t = (c1 - x)
            else:
                if normalize_t:
                    t = (x - c0) / width
                else:
                    t = (x - c0)

            if custom_function is not None:
                a = custom_function(t)
            elif use_sigmoid:
                a = 1.0 / (1.0 + np.exp(-k * (t - 0.5)))
            else:
                a = t ** power

            a = float(np.clip(a, 0.0, alpha_max))


            alpha[r0:r1, x] = np.maximum(alpha[r0:r1, x], a)

    if feather_sigma and feather_sigma > 0:
        alpha = skimage.filters.gaussian(alpha, sigma=feather_sigma)
        alpha = np.clip(alpha, 0.0, 1.0)

    out = (1 - alpha[..., None]) * image + alpha[..., None] * blurred
    return out



def crop_by_box(image_array, r0, r1, c0, c1, match_to_fig_size, dpi = 100):
    #bboxes = ia_edges_to_skimage_bboxes(self.fig, self.ax, self.ia_data, font_size_pt=self.font_size)
    #test_box_idx = [bboxes[i] for i in crop_idx]
    
    #r0 = min([test_box_idx[i]['r0'] for i in range(len(test_box_idx))])
    #r1 = max([test_box_idx[i]['r1'] for i in range(len(test_box_idx))])
    #c0 = min([test_box_idx[i]['c0'] for i in range(len(test_box_idx))])
    #c1 = max([test_box_idx[i]['c1'] for i in range(len(test_box_idx))])
    #image = mpl_to_skimage(self.fig)
    target_pixel_size = (match_to_fig_size[0] * dpi, match_to_fig_size[1] * dpi)
    if c1 - c0 > target_pixel_size[0]:
        c0 = c1 - target_pixel_size[0]
    image_float = skimage.img_as_float(image_array)
    cropped_image_float = image_float[r0:r1, c0:c1]
    
    #print("cropped size",cropped_image_float.shape)
    #print("target_pixel_size", target_pixel_size)
    pad_h = target_pixel_size[1] - cropped_image_float.shape[0]
    pad_w = target_pixel_size[0] - cropped_image_float.shape[1]

    cropped_image_float = np.pad(
        cropped_image_float,
        (
            (pad_h // 2, pad_h - pad_h // 2),
            (pad_w // 2, pad_w - pad_w // 2),
            (0, 0),
        ),
        mode="constant",
        constant_values=1.0,
    )
    return cropped_image_float
def main():
    import yaml
    path_to_yaml = "/home/lijunlin/llamagaze/paras/plot_paras.yaml"
    with open(path_to_yaml, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    
    plotter = GazePlotter(**data)
    ias = ["我","在","看星星"]
    lps = [10,10,10]
    durs = [300,400,500]
    position_2ds = [(0,i) for i in range(len(ias))]
    char_width = 28
    space_width = 0
    margin = 5
    y_base = 0
    ia_data = plotter.draw_ia_plot(ias,position_2ds,title = "wozaikanxingxing",show=False)
    fig, ax = plt.subplots(figsize=(plotter.figsize[0], plotter.figsize[1]))
    plot_on_ax(ax, ia_data, path_data=None, title = "test", 
            fontproperties=plotter.prop, 
            font_size=plotter.font_size, 
            xlim = plotter.xlim, 
            ylim = plotter.ylim,
            draw_box=False
            )
    

if __name__ == "__main__":
    main()