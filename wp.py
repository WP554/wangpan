import streamlit as st
import requests
import pandas as pd
import re
import jieba
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pygal
from pygal.style import LightColorizedStyle as LCS
import plotly.express as px
import altair as alt
import random
import validators


# URL fetch function
def fetch_content(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Check if the request was successful
    response.encoding = 'utf-8'
    return response.text


def remove_html_tags(html):
    """Remove HTML tags"""
    return re.sub(r'<[^>]+>', '', html)


def remove_punctuation_and_english(text):
    """Remove punctuation and English characters, keep Chinese characters"""
    return re.sub(r'[A-Za-z0-9\s+]|[^\u4e00-\u9fa5]+', '', text)


# Generate word cloud
def generate_wordcloud(word_counts):
    wordcloud = WordCloud(font_path='path/to/your/font.ttf', width=800, height=400,
                          background_color='white').generate_from_frequencies(word_counts)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Do not show the axes
    plt.tight_layout()
    return plt


# Visualization functions for Matplotlib
def plot_matplotlib(freq_df, chart_type):
    plt.figure(figsize=(12, 6))
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Example font that supports Chinese

    if chart_type == "柱状图":  # Bar chart
        plt.bar(freq_df['词语'], freq_df['频率'], color='skyblue')
        plt.title('柱状图')
        plt.xlabel('词语')
        plt.ylabel('频率')
        plt.xticks(rotation=45)

    elif chart_type == "折线图":  # Line chart
        plt.plot(freq_df['词语'], freq_df['频率'], marker='o', color='orange')
        plt.title('折线图')
        plt.xlabel('词语')
        plt.ylabel('频率')
        plt.xticks(rotation=45)

    elif chart_type == "饼图":  # Pie chart
        plt.pie(freq_df['频率'], labels=freq_df['词语'], autopct='%1.1f%%', startangle=90)
        plt.title('饼图')
        plt.axis('equal')

    elif chart_type == "面积图":  # Area chart
        plt.fill_between(freq_df['词语'], freq_df['频率'], color='lightgreen', alpha=0.5)
        plt.title('面积图')
        plt.xlabel('词语')
        plt.ylabel('频率')
        plt.xticks(rotation=45)

    elif chart_type == "散点图":  # Scatter plot
        plt.scatter(freq_df['词语'], freq_df['频率'], color='red')
        plt.title('散点图')
        plt.xlabel('词语')
        plt.ylabel('频率')
        plt.xticks(rotation=45)

    plt.tight_layout()
    return plt


# Visualization functions for Plotly
def plotly_visualizations(freq_df, chart_type):
    fig = None
    if chart_type == "柱状图":
        fig = px.bar(freq_df, x='词语', y='频率', title='Plotly柱状图', color='频率', text='频率')
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')  # Display values on top of bars
    elif chart_type == "折线图":
        fig = px.line(freq_df, x='词语', y='频率', title='Plotly折线图', markers=True)
    elif chart_type == "饼图":
        fig = px.pie(freq_df, names='词语', values='频率', title='Plotly饼图')
    elif chart_type == "面积图":
        fig = px.area(freq_df, x='词语', y='频率', title='Plotly面积图')
    elif chart_type == "散点图":
        fig = px.scatter(freq_df, x='词语', y='频率', title='Plotly散点图', color='频率')
    fig.update_layout(
        font=dict(family='Arial, "微软雅黑", "宋体", "SimHei"', size=14))  # Font setting to support Chinese
    return fig


# Visualization functions for Altair
def altair_visualizations(freq_df, chart_type):
    base = alt.Chart(freq_df).encode(
        x=alt.X('词语', title='词语'),
        y=alt.Y('频率', title='频率')
    ).properties(title='Altair Visualization', width=700, height=400)

    if chart_type == "柱状图":
        chart = base.mark_bar().encode(color='频率')
    elif chart_type == "折线图":
        chart = base.mark_line(point=True)
    elif chart_type == "饼图":
        chart = alt.Chart(freq_df).mark_arc().encode(
            theta=alt.Theta('频率'),
            color=alt.Color('词语', legend=None),
            tooltip=['词语', '频率']
        ).properties(title='Altair饼图')
    elif chart_type == "面积图":
        chart = base.mark_area()
    elif chart_type == "散点图":
        chart = base.mark_circle(size=60).encode(tooltip=['词语', '频率'])

    return chart


# Pygal visualizations with color
def pygal_visualizations(freq_df, chart_type):
    colors = ['#FF5733', '#33FF57', '#3357FF', '#F33FFF', '#FFB733', '#33FFF5', '#FF33D4', '#FF3333', '#57FF33',
              '#FFC733']

    if chart_type == "柱状图":
        bar_chart = pygal.Bar(style=LCS, explicit_size=True)
        bar_chart.title = 'Pygal柱状图'

        for index, row in freq_df.iterrows():
            # Get random color for each bar
            color = random.choice(colors)
            bar_chart.add(row['词语'], row['频率'], stroke=True, fill=True, style={"fill": color})

        return bar_chart

    elif chart_type == "折线图":
        line_chart = pygal.Line(style=LCS)
        line_chart.title = 'Pygal折线图'
        line_chart.x_labels = freq_df['词语'].tolist()
        line_chart.add('频率', freq_df['频率'].tolist())
        return line_chart

    elif chart_type == "饼图":
        pie_chart = pygal.Pie(style=LCS)
        pie_chart.title = 'Pygal饼图'
        for index, row in freq_df.iterrows():
            pie_chart.add(row['词语'], row['频率'])
        return pie_chart

    elif chart_type == "面积图":
        area_chart = pygal.Line(style=LCS)
        area_chart.title = 'Pygal面积图'
        area_chart.x_labels = freq_df['词语'].tolist()
        area_chart.add('频率', freq_df['频率'].tolist(), stroke_style={'width': 2, 'color': 'blue'})
        return area_chart

    elif chart_type == "散点图":
        scatter_chart = pygal.XY(style=LCS)
        scatter_chart.title = 'Pygal散点图'
        scatter_chart.add('频率', [(i, freq) for i, freq in zip(freq_df['词语'], freq_df['频率'])])
        return scatter_chart

    elif chart_type == "箱线图":
        box_chart = pygal.Box(style=LCS)
        box_chart.title = 'Pygal箱线图'
        box_chart.add('频率', freq_df['频率'].tolist())
        return box_chart


def main():
    st.title("文本分析与词频可视化")

    stopwords_file = st.sidebar.file_uploader("上传停用词文件 (stopwords.txt):", type=['txt'])
    url = st.sidebar.text_input("输入文章的 URL:", placeholder="https://example.com")

    # Create two dropdowns for library and chart selection
    visualization_type = st.sidebar.selectbox("选择可视化库:", ["Matplotlib", "Plotly", "Altair", "Pygal"])
    chart_type = st.sidebar.selectbox("选择图表类型:", ["柱状图", "折线图", "饼图", "面积图", "散点图", "箱线图"])

    if st.sidebar.button("抓取并分析"):
        if not validators.url(url):
            st.error("请输入有效的 URL!")
            return

        if stopwords_file is not None:
            stopwords = set(stopwords_file.read().decode('utf-8').splitlines())
        else:
            st.error("请上传停用词文件!")
            return

        try:
            html = fetch_content(url)
            clean_text = remove_html_tags(html)
            clean_text = remove_punctuation_and_english(clean_text)
            words = jieba.lcut(clean_text)

            meaningful_words = [word for word in words if word not in stopwords and len(word) > 1]
            word_counts = Counter(meaningful_words)
            most_common_words = word_counts.most_common(20)
            freq_df = pd.DataFrame(most_common_words, columns=['词语', '频率'])

            with st.expander("点击查看提取的文章"):
                st.write(clean_text)

            st.write("词频排名前 20 的词汇：")
            st.dataframe(freq_df)

            # Generate the corresponding visualizations based on user selection
            if chart_type == "词云图":
                st.write("生成词云图：")
                wordcloud_fig = generate_wordcloud(word_counts)
                st.pyplot(wordcloud_fig)  # Display word cloud
            else:
                if visualization_type == "Matplotlib":
                    fig = plot_matplotlib(freq_df, chart_type)
                    st.pyplot(fig)

                elif visualization_type == "Plotly":
                    fig = plotly_visualizations(freq_df, chart_type)
                    st.plotly_chart(fig)

                elif visualization_type == "Altair":
                    chart = altair_visualizations(freq_df, chart_type)
                    st.altair_chart(chart)

                elif visualization_type == "Pygal":
                    chart = pygal_visualizations(freq_df, chart_type)
                    st.write(chart.render(is_unicode=True), unsafe_allow_html=True)

        except Exception as e:
            st.error(f"发生错误：{e}")


if __name__ == "__main__":
    main()