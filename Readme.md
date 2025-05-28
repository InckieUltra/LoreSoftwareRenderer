# 真实感渲染大作业

---

CPU软件渲染器，支持Phong Shading光栅化渲染，以及漫反射或者镜面反射的光线追踪渲染。

使用SDL2创建窗口和接受交互输入，使用TinyObjLoader读取模型。

没做啥优化，光线追踪现在是1spp和1次弹射，时序降噪后看着还行。

---

## Showcase
时序降噪静帧：

![D3 VK3QYR{VTASDQ54YQT4J](https://github.com/user-attachments/assets/b0efe041-bf3a-4950-b465-6784ce332c3a)

镜面反射动态：
https://github.com/user-attachments/assets/3526a657-d034-4a8b-9fe1-0305c953f5d0

漫反射动态：
https://github.com/user-attachments/assets/fd73b6a6-6ecf-4d81-9874-b78293428ec4



---

## 完成度表格

| 功能点                     | 完成情况 | 备注             |
|----------------------------|----------|------------------|
| OBJ模型加载                |     DONE     |      TinyObjLoader            |
| 透视投影                   |    DONE      |                  |
| 相机漫游（旋转/平移/缩放） |     DONE     |        Blender风格交互          |
| 正交投影（附加）           |     DONE     |                  |
| 多模型渲染                 |    DONE      |                  |
| Phong光照模型              |     DONE     |                  |
| 光线追踪                   |     DONE     |                  |
| 纹理贴图                   |     DONE     |                  |
| 面光源（附加）             |          |                  |
| BRDF反射模型（附加）       |    DONE      |      简单的漫反射BRDF            |
| 透明纹理（附加）           |          |                  |
| 全局光照（附加）           |    DONE      |        光线一次反射          |


---


## 使用说明

1. **环境要求**  
   - cmake和一个cpp编译器

2. **运行方法**  
   - 编译项目
   - 鼠标中键旋转视角，Shift+鼠标中键移动相机位置，滚轮缩放

---

## 参考

- GPT
