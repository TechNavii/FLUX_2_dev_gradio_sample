from flux_app import launch_app

if __name__ == "__main__":
    # Forces the offline path (local text encoder). Ensure weights are downloaded/cached first.
    launch_app(use_remote_encoder=False)
