c = get_config()  # Get the config object.

c.NotebookApp.ip = '0.0.0.0'  # Serve notebooks locally.
c.NotebookApp.port = 8889
c.NotebookApp.open_browser = False  # Do not open a browser window by default when using notebooks.
c.NotebookApp.notebook_dir = '/workspace' 

