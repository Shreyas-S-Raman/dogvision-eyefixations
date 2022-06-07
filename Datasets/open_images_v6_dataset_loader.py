import fiftyone as fo
import fiftyone.zoo as fo_zoo

'''OpenImages V6 Classes: class name used in custom dataset (OpenImages classname):
1. bird, 2. duck, 3. hamster, 4. insect, 5. lizard, 6. rabbit, 7. raccoon, 8. squirrel, 9. dog, 10. cat
11. bicycle, 12. bus, 13. car, 14. skateboard, 15. train
16. human_eye (human eye), 17. human_foot (human foot), 18. human_head (human head), 19. human_hand (human hand)
20. flower, 21. plant, 22. tree
23. building, 24. house, 25. building (office building), 26. building_entrance (porch), 27. skyscraper
27. ball, 28. dog_toy (toy)'''

#downloading only the segmentation label data in the dataset
#class_list = ["Ball","Bus","Duck","Human Eye","Lizard","Raccoon","Train","Car","Flower","Human Foot","Office Building","Skateboard","Tree","Bicycle","Chicken","Hamster","Human Hand","Plant","Skyscraper","Bird","Deer","House","Human Head","Porch","Squirrel","Building","Dog Bed","Houseplant","Insect","Rabbit","Toy"]

class_name = input("Enter Label to Extract: ").strip()

dataset = fo_zoo.load_zoo_dataset('open-images-v6',label_types=['segmentations'], classes = [class_name], max_samples=500, shuffle=True, dataset_name="open-images-dog-fixations")

print(dataset.info['segmentation_classes'])
