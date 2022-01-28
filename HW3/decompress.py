import zlib

decompress_path = './CNN.tflite.zlib'
compressed_data = open(decompress_path, 'rb').read()
decompressed_data = zlib.decompress(compressed_data)

with open('./cnn.tflite', 'wb') as f:
    f.write(decompressed_data)