import streamlit as st
from PIL import Image, ImageDraw
import torch
import io
import streamlit as st

st.set_page_config(
    page_title="水果新鲜度识别",
    page_icon="🍎",
    layout="centered",
    initial_sidebar_state="collapsed"  # 手机端默认收起侧边栏
)

# 手机端样式优化
st.markdown("""
<style>
@media screen and (max-width: 768px) {
    .main .block-container {
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
    }
    h1 {
        font-size: 1.5rem !important;
    }
    h2 {
        font-size: 1.2rem !important;
    }
    h3 {
        font-size: 1rem !important;
    }
}
</style>
""", unsafe_allow_html=True)
# --- 页面配置 ---
st.set_page_config(page_title="水果新鲜度识别", page_icon="🍎", layout="centered")

# --- 标题 ---
st.title("🍎 水果新鲜度智能识别系统")
st.markdown("上传一张水果图片，AI 将自动分析其新鲜度。")

# --- 1. 加载模型 (直接使用本地库，无需联网) ---
from ultralytics import YOLO
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1' # 修复云端环境加载cv2报错
# 初始化模型对象，指向你的权重文件
model = YOLO('best.pt')

# --- 饮食搭配知识库（key 与模型类别对应） ---
food_pairing = {
    "apple": {
        "name": "苹果",
        "avoid": [
            "不宜与海鲜同食（果酸与蛋白质结合，可能引发腹痛）",
            "不宜与萝卜同食（可能抑制甲状腺功能）"
        ],
        "recommend": [
            "搭配蓝莓：果胶+花青素协同，有助于改善心血管健康",
            "搭配洋葱+茶叶：保护心脏，降低心脏病风险"
        ],
        "tip": "苹果带皮吃营养更丰富，饭前食用有助于控制食欲；常温可保存1-2周，冷藏可保存1-2个月"
    },
    "banana": {
        "name": "香蕉",
        "avoid": [
            "不宜与酸奶同食（果酸使蛋白质凝结，肠胃敏感者易腹胀）",
            "不宜与西瓜同食（两者皆偏寒凉，叠加易腹泻）"
        ],
        "recommend": [
            "搭配燕麦/奇亚籽：增强饱腹感，稳定血糖",
            "搭配牛奶：补充钾元素和蛋白质，适合运动后食用"
        ],
        "tip": "香蕉富含钾元素，建议运动后食用；常温保存即可，成熟后2-3天内吃完；不建议放冰箱（皮会变黑）"
    },
    "grape": {
        "name": "葡萄",
        "avoid": [
            "不宜与海鲜同食（果酸凝固蛋白质，刺激肠胃）",
            "不宜与萝卜同食（可能影响甲状腺功能）"
        ],
        "recommend": [
            "搭配枸杞：补血效果更佳",
            "搭配酸奶：花青素与益生菌协同抗氧化"
        ],
        "tip": "葡萄皮含花青素和白藜芦醇，建议带皮食用；表面白霜是天然蜡质保护层，洗掉会缩短保存期"
    },
    "guava": {
        "name": "番石榴",
        "avoid": [
            "不宜与辛辣食物同食（容易刺激肠胃）",
            "不宜过量食用（容易导致便秘）"
        ],
        "recommend": [
            "搭配柠檬汁：促进铁吸收，口感更清爽",
            "搭配酸奶：膳食纤维与益生菌协同，促进消化"
        ],
        "tip": "番石榴维C含量极高，是橙子的数倍；常温保存3-5天，切开后需冷藏并尽快食用"
    },
    "jujube": {
        "name": "枣",
        "avoid": [
            "不宜与黄瓜/胡萝卜同食（会破坏维C）",
            "不宜与动物肝脏同食（矿物质会氧化维C）"
        ],
        "recommend": [
            "搭配银耳：滋阴润肺，适合秋冬季节",
            "搭配枸杞：补气养血，增强免疫力"
        ],
        "tip": "鲜枣维C含量极高，但保质期短，建议冷藏保存并在3-7天内食用完毕"
    },
    "orange": {
        "name": "橙子",
        "avoid": [
            "不宜与牛奶同食（果酸使蛋白质凝结，建议间隔1小时）",
            "不宜与萝卜同食（萝卜中的酶会破坏维C）"
        ],
        "recommend": [
            "搭配蜂蜜：调理胃气不和、食欲不振",
            "搭配菠菜等绿叶蔬菜：维C促进植物铁吸收"
        ],
        "tip": "橙子富含维C，建议饭后1小时食用，避免空腹；室温下约可保存2周"
    },
    "pomegranate": {
        "name": "石榴",
        "avoid": [
            "不宜与海鲜同食（鞣酸与蛋白质结合，影响消化）",
            "不宜与螃蟹同食（可能引起肠胃不适）"
        ],
        "recommend": [
            "搭配酸奶：抗氧化效果更佳",
            "搭配蜂蜜：润肺止咳，口感更柔和"
        ],
        "tip": "石榴富含抗氧化物质，常温可保存1-2周，冷藏可延长至1个月；剥好的籽需密封冷藏"
    },
    "strawberry": {
        "name": "草莓",
        "avoid": [
            "不宜与钙剂同食（草酸与钙结合影响钙吸收）",
            "不宜与黄瓜同食（黄瓜中的酶会破坏维C）"
        ],
        "recommend": [
            "搭配牛奶：清凉解渴、养心安神",
            "搭配山楂：消食减脂，改善消化不良"
        ],
        "tip": "草莓富含维C，保质期短（3-7天），建议冷藏保存，食用前用淡盐水浸泡10分钟"
    }
}
# --- 2. 文件上传器 ---
uploaded_file = st.file_uploader("选择一张图片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 读取图片
    image = Image.open(uploaded_file)
    st.image(image, caption='上传的原图', use_column_width=True)

    # --- 3. 开始识别 ---
    if st.button('开始识别'):
        with st.spinner('AI 正在分析中，请稍候...'):
            # 使用模型进行推理
            # results 是一个包含所有检测信息的列表
            results = model(image)

            # --- 4. 处理并展示结果 ---
            # results[0] 是第一张（也是唯一一张）图片的结果
            result = results[0]

            # 获取带标注框的图片 (PIL Image 格式)
            # plot() 方法会自动画上框和标签
            annotated_img = result.plot()
            annotated_img_pil = Image.fromarray(annotated_img[..., ::-1])  # BGR to RGB

            st.image(annotated_img_pil, caption='识别结果', use_column_width=True)

            # 解析详细信息
            st.subheader("📊 详细分析")

            # =============================================
            # 模块：多水果食用顺序建议
            # =============================================

            # 1. 收集本次识别到的所有水果信息
            fruits_detected = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])

                # 提取水果 key 和状态
                fruit_key = label.replace("Fresh", "").replace("Rotten", "").lower().strip()
                is_rotten = "Rotten" in label

                # 获取中文名（如果知识库中有）
                fruit_name = food_pairing.get(fruit_key, {}).get("name", fruit_key)

                fruits_detected.append({
                    "label": label,
                    "fruit_key": fruit_key,
                    "name": fruit_name,
                    "is_rotten": is_rotten,
                    "conf": conf
                })

            # 2. 去重（同一种水果只保留一个）
            unique_fruits = {}
            for f in fruits_detected:
                key = f["fruit_key"]
                # 如果同一种水果出现多次，优先保留腐烂的那个（更需要提醒）
                if key not in unique_fruits or f["is_rotten"]:
                    unique_fruits[key] = f

            fruits_list = list(unique_fruits.values())

            # 3. 按优先级排序：腐烂 > 新鲜（腐烂的排最前面，提醒用户处理）
            # 排序规则：腐烂的排前面，新鲜的排后面
            fruits_list.sort(key=lambda x: (not x["is_rotten"], x["name"]))

            # 4. 展示食用顺序建议
            if len(fruits_list) > 1:
                st.markdown("---")
                st.markdown("### 🍽️ 食用顺序建议")
                st.markdown("根据识别结果，建议你按以下顺序食用：")

                for i, fruit in enumerate(fruits_list, 1):
                    if fruit["is_rotten"]:
                        st.markdown(f"**{i}. {fruit['name']}** — ⚠️ 已变质，请立即丢弃，不要食用！")
                    else:
                        # 根据水果种类给出不同的建议
                        tip = ""
                        if fruit["fruit_key"] == "strawberry":
                            tip = "保质期最短，建议今天吃掉"
                        elif fruit["fruit_key"] == "grape":
                            tip = "保质期较短，建议1-2天内食用"
                        elif fruit["fruit_key"] == "banana":
                            tip = "成熟后2-3天内食用最佳"
                        elif fruit["fruit_key"] == "jujube":
                            tip = "鲜枣保质期短，建议尽快食用"
                        elif fruit["fruit_key"] == "guava":
                            tip = "常温可保存3-5天"
                        elif fruit["fruit_key"] == "orange":
                            tip = "较耐存放，可放后面吃"
                        elif fruit["fruit_key"] == "apple":
                            tip = "很耐存放，可以最后吃"
                        elif fruit["fruit_key"] == "pomegranate":
                            tip = "较耐存放，可以最后吃"
                        else:
                            tip = "建议尽快食用"

                        st.markdown(f"**{i}. {fruit['name']}** — ✅ {tip}")

                # 额外提示
                rotten_count = sum(1 for f in fruits_list if f["is_rotten"])
                fresh_count = len(fruits_list) - rotten_count

                if rotten_count > 0:
                    st.warning(f"⚠️ 本次检测到 {rotten_count} 个已变质水果，建议立即清理，避免霉菌污染其他水果。")

                st.info(f"📌 本次共识别到 {len(fruits_list)} 种水果（{fresh_count} 个新鲜，{rotten_count} 个变质）")

            elif len(fruits_list) == 1:
                # 只识别到一种水果时，不显示排序，只显示单个水果的状态
                pass  # 单个水果的情况已经在上面的饮食搭配模块中展示了


            # 获取所有检测到的对象
            boxes = result.boxes
            if len(boxes) > 0:
                for i, box in enumerate(boxes):
                    # 获取类别、置信度
                    cls_id = int(box.cls[0])  # 类别ID
                    conf = box.conf[0]  # 置信度
                    label = result.names[cls_id]  # 类别名称

                    # 这里可以加入你自己的逻辑来判断新鲜度
                    # 例如，如果置信度高于0.8，就认为是“新鲜”
                    freshness = "新鲜" if conf > 0.8 else "一般"
                    color = "green" if conf > 0.8 else "orange"

                    st.markdown(f"""
                    **检测结果 {i + 1}:**
                    - **类别:** {label}
                    - **置信度:** {conf:.2f}
                    - **新鲜度评估:** <span style='color:{color}'>**{freshness}**</span>
                    """, unsafe_allow_html=True)
                    st.divider()

                    # --- 在显示完类别和置信度之后，添加饮食搭配建议 ---
                    # 从模型输出的类别名中提取水果 key
                    # 例如：FreshApple -> apple, RottenBanana -> banana
                    fruit_key = label.replace("Fresh", "").replace("Rotten", "").lower().strip()
                    if fruit_key in food_pairing:
                        info = food_pairing[fruit_key]
                        fruit_name = info["name"]

                        # 判断是新鲜还是腐烂
                        is_rotten = "Rotten" in label

                        st.markdown("---")

                        if is_rotten:
                            st.warning(
                                f"⚠️ **{fruit_name}** 已变质，**请勿食用！** 霉菌毒素已扩散至整个果实，切掉霉斑也无法去除。")
                        else:
                            st.success(f"✅ **{fruit_name}** 状态良好，可以放心食用")

                            # 不宜同食
                            st.markdown("#### 🚫 不宜同食")
                            for item in info["avoid"]:
                                st.markdown(f"- ⚠️ {item}")

                            # 推荐搭配
                            st.markdown("#### ✅ 推荐搭配")
                            for item in info["recommend"]:
                                st.markdown(f"- 💡 {item}")

                            # 保鲜小贴士
                            st.markdown(f"#### 🧊 保鲜小贴士")
                            st.info(info["tip"])
                    else:
                        st.markdown("#### ℹ️ 暂无该水果的饮食搭配信息")
            else:
                st.info("未检测到任何水果。")

else:
    st.info("请先上传图片。")


