class Config:
    # Paths
    disaster_image_dir = 'Disaster Image Dataset/'
    medic_dir = 'MEDIC Dataset/'
    era5_dir = 'ERA5 Meteorological Raster Dataset/'
    news_dir = 'Environmental News Dataset/'
    delhi_urban_dir = 'Delhi Urban Risk Dataset/'

    # Model
    text_dim = 768
    image_dim = 512
    geo_dim = 256
    fusion_dim = 512
    attn_heads = 4
    attn_layers = 2
    dropout = 0.2

    # Training
    learning_rate = 1e-4
    batch_size = 32
    epochs = 50
    weight_decay = 0.01
    device = 'cuda'  # or 'cpu' 