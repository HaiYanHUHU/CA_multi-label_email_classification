from data import get_input_data, remove_noise






if __name__ == '__main__':
    df = get_input_data(['data/AppGallery.csv', 'data/Purchasing.csv'])
    print(remove_noise(df))