---
title: Pyecharts 不同颜色绘制正负柱状图
date: 2021-09-27 10:52:10
index_img: /img/article/pyecharts.jpg
categories:
    - Python
tags:
    - Pyecharts
comment: 'valine'
---
## 如题
<!-- more -->
```
import akshare as ak
import pyecharts.options as opts
from pyecharts.charts import Bar, Line
from pyecharts.commons.utils import JsCode

fund_em_info_df = ak.fund_em_open_fund_info(fund="006008", indicator="单位净值走势")

fund_name = '诺安积极配置混合C'
x_data = fund_em_info_df['净值日期'].tolist()
y_data = fund_em_info_df['单位净值'].tolist()
z_data = fund_em_info_df['日增长率'].tolist()

background_color_js = (
    "new echarts.graphic.LinearGradient(0, 0, 0, 1, "
    "[{offset: 0, color: '#c86589'}, {offset: 1, color: '#06a7ff'}], false)"
)
area_color_js = (
    "new echarts.graphic.LinearGradient(0, 0, 0, 1, "
    "[{offset: 0, color: '#eb64fb'}, {offset: 1, color: '#3fbbff0d'}], false)"
)


bar = (
    Bar(init_opts=opts.InitOpts(bg_color=JsCode(background_color_js), width='700px', height='450px'))     ## width, height修改画布大小
    .add_xaxis(xaxis_data=x_data)
    .add_yaxis(
        series_name="",
        y_axis=z_data,
        label_opts=opts.LabelOpts(is_show=False),
        itemstyle_opts=opts.ItemStyleOpts(
            ### 调用js代码绘制不同颜色
            color=JsCode(
                """
                    function(params) {
                        var colorList;
                        if (params.data >= 0) {
                          colorList = '#FF4500';
                        } else {
                          colorList = '#14b143';
                        }
                        return colorList;
                    }
                    """
                )
            )
        )
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title=fund_name,
            pos_bottom="90%",
            pos_left="center",
            title_textstyle_opts=opts.TextStyleOpts(color="#fff", font_size=16),
        ),
        xaxis_opts=opts.AxisOpts(
            type_="category",
            boundary_gap=False,
            axislabel_opts=opts.LabelOpts(margin=30, color="#ffffff63"),
            axisline_opts=opts.AxisLineOpts(is_show=False),
            axistick_opts=opts.AxisTickOpts(
                is_show=True,
                length=25,
                linestyle_opts=opts.LineStyleOpts(color="#ffffff1f"),
            ),
            splitline_opts=opts.SplitLineOpts(
                is_show=True, linestyle_opts=opts.LineStyleOpts(color="#ffffff1f")
            ),
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            position="left",
            axislabel_opts=opts.LabelOpts(margin=20, color="#ffffff63"),
            axisline_opts=opts.AxisLineOpts(
                linestyle_opts=opts.LineStyleOpts(width=2, color="#fff")
            ),
            axistick_opts=opts.AxisTickOpts(
                is_show=True,
                length=15,
                linestyle_opts=opts.LineStyleOpts(color="#ffffff1f"),
            ),
            splitline_opts=opts.SplitLineOpts(
                is_show=True, linestyle_opts=opts.LineStyleOpts(color="#ffffff1f")
            ),
        ),
#         legend_opts=opts.LegendOpts(is_show=True),
        datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")]    ## 时间轴显示并可同通过鼠标滑动
    )
)


line = (
    Line(init_opts=opts.InitOpts(bg_color=JsCode(background_color_js)))
    .add_xaxis(xaxis_data=x_data)
    .add_yaxis(
        series_name="",
        y_axis=[round(i * 10, 2) for i in y_data],
        is_smooth=True,
        is_symbol_show=True,
        symbol="circle",
        symbol_size=6,
        linestyle_opts=opts.LineStyleOpts(color="#fff"),
        label_opts=opts.LabelOpts(is_show=True, position="top", color="white"),
        itemstyle_opts=opts.ItemStyleOpts(
            color="red", border_color="#fff", border_width=3
        ),
        tooltip_opts=opts.TooltipOpts(is_show=False),
        areastyle_opts=opts.AreaStyleOpts(color=JsCode(area_color_js), opacity=1),
    )
)

bar.overlap(line)        ## 混合柱状图和线图
bar.render_notebook()

```
结果如下
<iframe src="/img/bar_line.html" width="100%" height="500" name="topFrame" scrolling="yes"  noresize="noresize" frameborder="0" id="topFrame"></iframe>

参考
* https://gallery.pyecharts.org/#/Candlestick/professional_kline_chart
