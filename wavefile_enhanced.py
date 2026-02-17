    def load_svd_model(self):
        try:
            self.svd_pipeline = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
            )
            if torch.cuda.is_available():
                self.svd_pipeline.enable_model_cpu_offload()
                print("SVD model loaded (GPU mode with offload)")
            else:
                print("SVD model loaded (CPU mode - slow)")
        except Exception as e:
            print(f"Failed to load SVD model: {e}")
            import traceback
            traceback.print_exc()