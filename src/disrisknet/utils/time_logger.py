import time
import os

class time_logger():
    def __init__(self, time_logger_step, hierachy=1, model_name="MODEL_NAME", logger_dir='../..logs', log_name=''):
        if time_logger_step == 0:
            self.logQ = False
        else:
            self.logQ = True
        self.time_logger_step = time_logger_step
        self.step_count = 0
        self.hierachy = hierachy
        self.time = time.time()
        self.model_name = model_name
        self.logger_dir = logger_dir
        self.log_name = log_name
        self.logger_stem = os.path.join(self.logger_dir, self.log_name)
        self.logger_path = '{}.monitor.log'.format(self.logger_stem)

    def log(self, s):
        if self.logQ and (self.step_count%self.time_logger_step==0):
            print("#" * 4 * self.hierachy, " ", s, "  --time elapsed: %.2f" % (time.time() - self.time))
            with open(self.logger_path, 'a') as f:
                f.write(" ".join([self.model_name, "---", "#" * 4 * self.hierachy, s,
                                  " --time elapsed: %.2f" % (time.time() - self.time), '\n']))
            f.close()
            self.time = time.time()

    def update(self):
        self.step_count += 1
        if self.logQ:
            self.log("#Refresh logger")

    def newline(self):
        if self.logQ:
            print('\n')
            with open(self.logger_path, 'a') as f:
                f.write('\n')
            f.close()
