import h5py

if __name__ == '__main__':
    file_path = '/path/to/your/file.h5'

    try:
        with h5py.File(file_path, 'r') as f:
            print(f.keys())
        print("Data is completely read from the file")
    except OSError as e:
        print(f"Error: {e}")