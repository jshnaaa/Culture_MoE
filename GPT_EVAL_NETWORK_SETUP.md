# GPT评测网络配置指南

## 🚨 常见问题：API调用超时

如果遇到以下错误：
```
⚠️ API调用失败: Request timed out.
❌ API调用失败，已达最大重试次数: Connection error.
```

**原因**：服务器无法访问OpenAI API（`api.openai.com`）

**解决方案**：配置代理或使用中转服务

---

## 🔧 解决方案

### 方案1：配置HTTP代理（推荐）

如果你有HTTP代理服务：

```bash
# 设置代理环境变量
export HTTP_PROXY=http://your-proxy-server:port
export HTTPS_PROXY=http://your-proxy-server:port

# 然后运行评测
bash run_eval_gpt.sh sk-proj-xxx 2 4o 20
```

**示例**：
```bash
# 假设代理地址是 127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

bash run_eval_gpt.sh sk-proj-xxx
```

### 方案2：使用OpenAI API中转服务

如果你有OpenAI API中转服务（国内可访问的代理）：

```bash
# 设置自定义API地址
export OPENAI_BASE_URL=https://your-proxy-domain.com/v1

# 然后运行评测
bash run_eval_gpt.sh sk-proj-xxx 2 4o 20
```

**常见中转服务**：
- API2D: `https://openai.api2d.net/v1`
- OpenAI-SB: `https://api.openai-sb.com/v1`
- 其他第三方中转服务

**注意**：使用中转服务时，API KEY可能需要使用中转服务提供的专用KEY。

### 方案3：增加超时时间（仅缓解，不解决根本问题）

```bash
# 设置更长的超时时间（默认60秒）
export OPENAI_TIMEOUT=120

bash run_eval_gpt.sh sk-proj-xxx 2 4o 20
```

---

## 🧪 测试网络连通性

在运行评测前，可以测试OpenAI API连通性：

### 测试1：直接访问（无代理）

```bash
curl -I https://api.openai.com/v1/models
```

**期望结果**：
- ✅ 成功：返回 HTTP 401（需要认证，但说明能连接）
- ❌ 失败：超时或连接错误

### 测试2：通过代理访问

```bash
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port

curl -I https://api.openai.com/v1/models
```

### 测试3：完整API调用测试

```bash
export OPENAI_API_KEY=sk-proj-your-key

curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

**期望结果**：返回可用模型列表的JSON

---

## 📝 完整配置示例

### 示例1：使用代理

```bash
#!/bin/bash

# 1. 配置代理
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

# 2. 运行评测
bash run_eval_gpt.sh sk-proj-your-api-key 2 4o 20
```

### 示例2：使用中转服务

```bash
#!/bin/bash

# 1. 配置中转服务
export OPENAI_BASE_URL=https://api.openai-sb.com/v1

# 2. 运行评测（使用中转服务的API KEY）
bash run_eval_gpt.sh sk-中转服务的KEY 2 4o 20
```

### 示例3：组合配置

```bash
#!/bin/bash

# 1. 配置代理
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

# 2. 配置超时
export OPENAI_TIMEOUT=120

# 3. 可选：使用中转服务
# export OPENAI_BASE_URL=https://your-proxy.com/v1

# 4. 运行评测
bash run_eval_gpt.sh sk-proj-your-api-key 2 4o 20
```

---

## 🔍 故障排查

### 问题1：代理配置后仍然超时

**可能原因**：
1. 代理服务器地址错误
2. 代理服务器未运行
3. 代理服务器不支持HTTPS

**解决方案**：
```bash
# 测试代理是否工作
curl -x http://your-proxy:port https://www.google.com

# 如果失败，检查代理服务状态
```

### 问题2：中转服务返回401错误

**原因**：API KEY不匹配

**解决方案**：
- 确认使用的是中转服务提供的专用KEY
- 或者使用原始OpenAI KEY（取决于中转服务支持）

### 问题3：部分请求成功，部分失败

**原因**：网络不稳定或API限流

**解决方案**：
```bash
# 增加重试次数和延迟
# 修改 eval_gpt.py 中的参数：
# --max_retries 5
# --retry_delay 5.0
```

---

## 💡 推荐配置

### 对于中国大陆服务器

**推荐方案**：代理 + 中转服务组合

```bash
# 1. 使用本地代理连接到境外服务器
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

# 2. 使用稳定的中转服务（可选，提高稳定性）
export OPENAI_BASE_URL=https://api.openai-sb.com/v1

# 3. 增加超时时间（应对网络波动）
export OPENAI_TIMEOUT=90

# 4. 运行评测
bash run_eval_gpt.sh sk-proj-xxx 2 4o 20
```

### 对于境外服务器

**无需配置**：直接运行即可

```bash
bash run_eval_gpt.sh sk-proj-xxx 2 4o 20
```

---

## 📚 环境变量参考

| 环境变量 | 说明 | 默认值 | 示例 |
|---------|------|--------|------|
| `HTTP_PROXY` | HTTP代理地址 | 无 | `http://127.0.0.1:7890` |
| `HTTPS_PROXY` | HTTPS代理地址 | 无 | `http://127.0.0.1:7890` |
| `OPENAI_BASE_URL` | 自定义API地址 | `https://api.openai.com/v1` | `https://api.openai-sb.com/v1` |
| `OPENAI_TIMEOUT` | 请求超时时间（秒） | `60.0` | `120` |
| `OPENAI_API_KEY` | OpenAI API密钥 | 无 | `sk-proj-xxx` |

---

## ⚠️ 安全提示

1. **代理安全**：确保代理服务器可信
2. **中转服务**：选择可信的中转服务提供商
3. **API KEY**：不要在中转服务中使用重要的OpenAI账户
4. **日志安全**：代理可能记录请求内容，注意数据隐私

---

## 📞 获取帮助

如果仍然无法解决，请检查：

1. **网络连通性**：
   ```bash
   ping api.openai.com
   ```

2. **DNS解析**：
   ```bash
   nslookup api.openai.com
   ```

3. **防火墙规则**：
   ```bash
   # 检查是否有防火墙阻止
   telnet api.openai.com 443
   ```

4. **Python依赖**：
   ```bash
   pip list | grep -E "openai|httpx"
   ```

确保安装了正确的版本：
```bash
pip install --upgrade openai httpx
```
