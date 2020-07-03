import math
key =1

def myinput():
    choice = input('请选择计算类型：（1-人力计算，2-工时计算）')
    if choice == '1':
        size = float(input('请输入项目大小：（1代表标准大小，请输入小数）'))
        number = None
        time = float(input('请输入工时数量：（请输入小数）'))
        return size,number,time
    elif choice == '2':
        size = float(input('请输入项目大小：（1代表标准大小，请输入小数）'))
        number = int(input('请输入人力数量：（请输入整数）'))
        time = None
        return size,number,time


def estimated(my_input):
    size = my_input[0]
    number = my_input[1]
    time = my_input[2]
    if (number == None) and (time != None):
        number = math.ceil(size * 80 / time)
        print('项目大小为%.1f个标准项目，如果需要在%.1f个工时完成，则需要人力数量为：%d人' %(size,time,number))
    elif (number != None) and (time == None):
        time = size * 80 / number
        print('项目大小为%.1f个标准项目，使用%d个人力完成，则需要工时数量为：%.1f个' %(size,number,time))


def agian():
    global key
    a = input("选择是否持续计算,选择持续计算输入是，选择停止持续输入输入否:\n")
    if a == '否':
        key = 0


def main():
    print('欢迎使用工作量计算小程序')
    while key == 1:
        my_input = myinput()
        estimated(my_input)
        agian()
    print('感谢使用工作量计算小程序')


main()


class Robert:
    def __init__(self, Robert_name, user_name):
        self.Robert_name = Robert_name
        self.user_name = user_name

    def again(self):
        a = input('请说出你得愿望')
        for i in range(3):
             print(a)
    def Hello(self):
        print('你好{},我是{}'.format(self.user_name,self.Robert_name))

robert = Robert('瓦力', '吴枫')
robert.Hello()
robert.again()
