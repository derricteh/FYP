For this challenge, we are providing you with ~64k images classified as advertisements.  The task is to predict a correct action-reason statement for the image regarding what the user should do and why they should do it; more information about this can be found below.  In terms of the layout of the dataset:

./train/QA_Combined_Action_Reason_train.json :
	In this file, there is information about 51,223 images.  For each item in the dictionary, you are given a list of two lists.  The first list contains true action-reason annotations from human annotators about this image.  The second list contains exactly fifteen action-reason annotations, consisting of all the true, human annotations as well as false annotations belonging to other images.

./test/QA_Combined_Action_Reason_test.json :
	In this file, there is information on 12,805 images.  Each image id is mapped to a list.  Each list contains exactly fifteen action-reason annotations.  Your goal is to predict one of the correct annotations out of the fifteen.  Since there are fifteen annotations given per image, you must guess one of the several, usually 3, correct annotations.  Please read the note below.  Since there are typically 3 correct annotations out of the 15 possible, you are assumed to have achieved at least a 20% accuracy by simply guessing.

NOTE: There are 64,028 total images.  We tried to maintain three annotations per user, but some images have more than three and some have fewer than three.  Regardless, the overwhelming majority of images possess three annotations.  Statistics can be found below:

Under 3 Annotations: 1,630 (2.54%)
Over 3 Annotations: 5,670 (8.84%)
Exactly 3 Annotations: 56,840 (88.62%)
Minimum Number of Annotations for an Image: 1
Maximum Number of Annotations for an Image: 5

NOTE: Information about the rest of the data comprising of slogans, sentiments, topics, and strategies can be found below.

===== INFORMATION REGARDING THE DATASET =====

Image Dataset

Images:

Our image dataset contains a total of 64,028 advertisement images verified by human annotators on Amazon Mechanical Turk. Images are split into 11 folders (subfolder 0 to subfolder 10 in each respective data split folder (i.e. "train")). For example, they are as follows:

./train_images/0/
./train_images/1/
...
./train_images/10/

and depending on the current version of the challenge, you'll either receive the tiny_test or test folder:

./test_images/0/
./test_images/1/
...
./test_images/10/

Annotation Files:

The annotation files are in json format.

Annotation files can be found in ./train/, and either ./test/ or ./tiny_test/ depending on the current version of the challenge.
The key in the file is the relative path ({subfolder}/{image_name})
The value is a list of annotations from different annotators. For each image, we posed the same question to 3-5 different annotators. We show all obtained annotations in this dataset.

For Q/A (action and reason), we provide unprocessed annotations as free-form sentences. For example:
{
  ...,
  "7/62717.jpg": ["I should do this because they are a patriotic company.",
                  "I should buy this because it is a fun, American company; a staple.",
                  "You should believe them because it is the American spirit."],
  ...
}

The file QA_Combined_Action_Reason.json contains results for the study in which the annotators gave a single combined answer to the "What should I do" and "Why should I do it" questions.

For "Slogans", we provide unprocessed annotations as free-form sentences. For example:
{
  ...,
  "7/62717.jpg": ["Because they are a patriotic company.",
                  "Because it is a fun, American company; a staple.",
                  "Because it is the American spirit."],
  ...
}

For "Topics", "Sentiments", and "Strategies", we show the class ID in the annotation files, and a corresponding mapping is provided in a separate txt file (Topics_List.txt, Sentiments_List.txt, and Strategies_List, respectively). The class ID identifies the particular topic, sentiment, or strategy that the annotator selected.

For example, for topics, if the value in the key-value pair is a list ["10", "11", "10"], this means that two annotators selected topic number 10 ("Electronics") and one selected topic number 11 ("Phone"). 
If the list contains strings, then some annotator(s) selected the "Other" option, and the string is the text that the annotator(s) entered after selecting the "Other" option. For example:
{
  ...,
  '9/91899.jpg': ['10', '10', 'musical instruments'],
  ...
}
 
Note that for "Sentiments" and "Strategies", one annotator could select multiple categories, so the value in the key-value pair is a list of lists. For example:
{
  ...,
   '3/40803.jpg': [['18', '19'],
                   ['19'],
                   ['18', '19', '25']],
  ...
}

IMPORTANT: A list of topics and sentiments can be found in topics_list.txt and sentiments_list.txt .  Both are included in the ZIP file.

For "Symbols", the value in the key-value pair is a list of annotations from multiple annotators, and each annotation is itself a list. The first 4 elements of that list are the coordinates of the bounding box that the annotator drew, and the last element is the phrase that annotator provided as a symbol label for that box (i.e. the "signified", using terminology from our paper). The coordinates x, y range from 0 to 500 as each image was scaled to 501x501. For example:
{
  ...,
   u'9/86039.jpg': [[356.0, 318.0, 230.0, 198.0, 'Dangerous'],
                    [212.0, 200.0, 417.0, 376.0, 'wild/adventurous'],
                    [224.0, 193.0, 446.0, 436.0, 'Adventure'],
                    [252.0, 236.0, 376.0, 335.0, 'Risk/Thrill'],
                    [259.0, 224.0, 387.0, 372.0, 'skydiving']],
  ...
}

Citation:

If you use our data in your paper, please cite the following paper:

Automatic Understanding of Image and Video Advertisements. Zaeem Hussain, Mingda Zhang, Xiaozhong Zhang, Keren Ye, Christopher Thomas, Zuha Agha, Nathan Ong, Adriana Kovashka. To appear, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), July 2017.