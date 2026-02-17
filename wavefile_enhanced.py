    def load_svd_model(self):
        try:
            self.svd_pipeline = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float32
            )
            # No offload for CPU
            print("SVD model loaded (CPU mode)")
        except Exception as e:
            print(f"Failed to load SVD model: {e}")