from flux_app import launch_app

if __name__ == "__main__":
    # Forces the remote text encoder path (internet required).
    launch_app(use_remote_encoder=True)
