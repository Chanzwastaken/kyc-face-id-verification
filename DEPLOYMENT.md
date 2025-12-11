# 🚀 Deployment Guide

This guide will help you deploy the TrustID KYC Face Verification application to the cloud.

---

## ⭐ Recommended: Streamlit Community Cloud (FREE)

The easiest way to deploy your Streamlit app with zero configuration.

### Prerequisites
- ✅ GitHub repository (you already have this!)
- ✅ GitHub account
- ✅ `requirements.txt` file (already included)

### Step-by-Step Instructions

#### 1. Ensure Your Code is Pushed to GitHub

```bash
# Check current status
git status

# If you have uncommitted changes, commit them
git add .
git commit -m "Prepare for deployment"
git push origin main
```

#### 2. Sign Up for Streamlit Community Cloud

1. Go to: **https://share.streamlit.io/**
2. Click **"Sign up"** or **"Continue with GitHub"**
3. Authorize Streamlit to access your GitHub repositories

#### 3. Deploy Your App

1. Click **"New app"** button
2. Fill in the deployment form:
   - **Repository:** `Chanzwastaken/kyc-face-id-verification`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** Choose a custom URL (e.g., `trustid-kyc-verification`)

3. Click **"Deploy!"**

#### 4. Wait for Deployment

- Initial deployment takes **2-5 minutes**
- Streamlit will automatically:
  - Install dependencies from `requirements.txt`
  - Download model weights (~100MB) on first run
  - Start your application

#### 5. Access Your App

Once deployed, you'll get a URL like:
```
https://trustid-kyc-verification.streamlit.app
```

---

## ⚠️ Important Considerations for Streamlit Cloud

### Resource Limitations (Free Tier)

- **RAM:** 1 GB
- **CPU:** 1 core (shared)
- **Storage:** Limited to app files + cached models
- **Timeout:** Apps sleep after inactivity

### Optimization Tips

1. **Model Caching:** Already implemented with `@st.cache_resource`
2. **Image Size:** Consider adding image compression for large uploads
3. **Cold Start:** First load may be slow due to model download

### Known Issues & Solutions

| Issue | Solution |
|-------|----------|
| App runs out of memory | Reduce image resolution before processing |
| Slow first load | Expected - models are being downloaded (~100MB) |
| App goes to sleep | Normal behavior - wakes up on next visit |

---

## 🎯 Alternative Deployment Options

### Option 2: Hugging Face Spaces (FREE + GPU)

Better for ML apps with GPU support.

**Steps:**

1. Create account: https://huggingface.co/
2. Go to: https://huggingface.co/spaces
3. Click **"Create new Space"**
4. Choose **"Streamlit"** as SDK
5. Upload your files: `app.py`, `requirements.txt`, `README.md`
6. Your app will be available at: `https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME`

**Advantages:**
- Free GPU access (optional)
- Better for ML workloads
- Larger resource limits

---

### Option 3: Railway.app (PAID - $5/month)

Production-grade deployment with better resources.

**Steps:**

1. Sign up: https://railway.app/
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose your repository
5. Railway auto-detects Streamlit and deploys

**Advantages:**
- More RAM (8GB+)
- Better performance
- Custom domains
- No sleep mode

**Cost:** ~$5-10/month depending on usage

---

### Option 4: Docker + Cloud Provider (ADVANCED)

For full control and enterprise deployment.

**Providers:**
- AWS (EC2, ECS, Lambda)
- Google Cloud Platform (Cloud Run, Compute Engine)
- Microsoft Azure (App Service, Container Instances)
- DigitalOcean (App Platform, Droplets)

**Steps:**

1. Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. Build and push to container registry
3. Deploy to your chosen cloud provider

---

## 🔧 Troubleshooting

### Deployment Fails

**Check:**
- All dependencies are in `requirements.txt`
- No syntax errors in `app.py`
- Python version compatibility (3.8-3.11)

### App Crashes on Streamlit Cloud

**Common causes:**
- Out of memory (1GB limit)
- Model download timeout
- Missing dependencies

**Solutions:**
- Reduce image resolution
- Use smaller models
- Add retry logic for model downloads

### Model Download Fails

**Error:** Network timeout or connection issues

**Solution:**
Add this to your app (already implemented):
```python
try:
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
except Exception as e:
    st.error("Network error downloading models. Please check internet connection.")
```

---

## 📊 Monitoring Your Deployed App

### Streamlit Cloud Dashboard

- View app logs
- Monitor resource usage
- Restart app if needed
- Update from GitHub automatically

### Analytics (Optional)

Add Google Analytics or Plausible to track:
- Number of verifications
- User engagement
- Performance metrics

---

## 🎉 Next Steps After Deployment

1. **Test the deployed app** with sample images
2. **Share the URL** on your portfolio/LinkedIn
3. **Monitor performance** in the first few days
4. **Gather feedback** from users
5. **Iterate and improve** based on usage

---

## 📝 Deployment Checklist

- [ ] Code pushed to GitHub
- [ ] `requirements.txt` is up to date
- [ ] Streamlit Cloud account created
- [ ] App deployed successfully
- [ ] Tested with sample images
- [ ] URL shared on portfolio
- [ ] README.md updated with live demo link

---

## 🔗 Useful Links

- **Streamlit Cloud Docs:** https://docs.streamlit.io/streamlit-community-cloud
- **Hugging Face Spaces:** https://huggingface.co/docs/hub/spaces
- **Railway Docs:** https://docs.railway.app/
- **Streamlit Forums:** https://discuss.streamlit.io/

---

## 💡 Pro Tips

1. **Add a live demo link** to your README.md
2. **Create a demo video** showing the verification process
3. **Add error handling** for edge cases
4. **Monitor costs** if using paid services
5. **Keep your repo public** for portfolio visibility

---

**Ready to deploy? Start with Streamlit Community Cloud - it's free and takes just 5 minutes!** 🚀
