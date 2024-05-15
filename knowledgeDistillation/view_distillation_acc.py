import matplotlib.pyplot as plt
import pandas as pd
from train_model import save_dir

<<<<<<< HEAD
=======

>>>>>>> 9633feafb276d18ae44feceff791806f83d50d31
def view_distillation_acc_st(student_acc_dir, teacher_acc_dir, student_noTea_acc_dir):
    '''对比学生模型, 教师模型, 以及无教师蒸馏的学生模型的acc曲线'''
    student_df = pd.read_csv(student_acc_dir)
    teacher_df = pd.read_csv(teacher_acc_dir)
    student_noTea_df = pd.read_csv(student_noTea_acc_dir)

<<<<<<< HEAD
    plt.figure(figsize = (5,5))
    plt.plot(student_df['epoch'], student_df['val_acc'], 'r', label = 'Student Acc')
    plt.plot(teacher_df['epoch'], teacher_df['val_acc'], 'b', label = 'Teacher Acc')
    plt.plot(student_noTea_df['epoch'], student_noTea_df['val_acc'], 'k', label = 'Student(Without Teacher) Acc')
=======
    plt.figure(figsize=(5, 5))
    plt.plot(student_df['epoch'], student_df['val_acc'], 'r', label='Student Acc')
    plt.plot(teacher_df['epoch'], teacher_df['val_acc'], 'b', label='Teacher Acc')
    plt.plot(student_noTea_df['epoch'], student_noTea_df['val_acc'], 'k', label='Student(Without Teacher) Acc')
>>>>>>> 9633feafb276d18ae44feceff791806f83d50d31
    plt.title('Comparing Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

<<<<<<< HEAD
=======

>>>>>>> 9633feafb276d18ae44feceff791806f83d50d31
if __name__ == '__main__':
    '''直接运行该文件,可查看对比的acc曲线'''
    student_acc_dir = save_dir + '/acc_and_loss_S.csv'
    teacher_acc_dir = save_dir + '/acc_and_loss_T.csv'
    student_noTea_acc_dir = save_dir + '/acc_and_loss_SnoT.csv'
    view_distillation_acc_st(student_acc_dir, teacher_acc_dir, student_noTea_acc_dir)
