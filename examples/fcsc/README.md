
### making dataset
```
./annotated_to_dataset.py # after making datasets of *.jpg and *.json by labelme
```

### training
```
./train_fcn32s.py -g 0
```

### inference
```
python infer.py -g 0 -m logs/20180307_215136/fcn32s_fcsc_for_73B1.npz -i ~/haraduka/20180307_imagedataset/raw/frame0000.jpg -o hoge
```

### launch
```
roslaunch jsk_perception fcn_object_segmentation.launch INPUT_IMAGE:=/camera/rgb/image_raw
```
