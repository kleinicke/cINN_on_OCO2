from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import tensorflow as tf
import io

#tensorboard --logdir=tmp/log_dir

import config as c
dc = c.dc
"""Visualization of the trainingprocess using tensorboard for pytorch
"""

n_imgs = 4
n_plots = 3
figsize = (4,4)
loss_names = ["loss"]
stat_names = ["meanabdiffs","meadabdiffs","offsets_den","offsets_median_den","param_differences"]

class Visualizer:
    def __init__(self, loss_labels,stat_labels):
            #self.n_losses = len(loss_labels)
            self.loss_labels = loss_labels
            self.stat_labels = stat_labels
            #self.counter = 0

            header = 'Epoch'
            for l in loss_labels:
                header += '\t\t%s' % (l)

            self.config_str = ""
            self.config_str += "==="*30 + "\n"
            self.config_str += "Config options:\n\n"

            for v in dir(c):
                if v[0]=='_': continue
                s=eval('c.%s'%(v))
                self.config_str += "  {:25}\t{}\n".format(v,s)

            self.config_str += "==="*30 + "\n"

            print(self.config_str)
            print(header)

    def update_losses(self, loss, *args):
        #line = '\r%.3i' % (self.counter)
        #self.counter += 1
        pass

    def update_offsets(self, offsets, *args):
        #line = '\r%.3i' % (self.counter)
        #for l in offsets:
        #    line += '\t\t%.4f' % (l)
        #
        #self.counter += 1
        pass

    def update_stats(self, *args):
        pass

    def update_std(self, *args):
        pass
        

    def make_step(self, step_size = 1, *args):
        pass


    def create_graph(self, *args):
        pass

    def update_running(self, *args):
        pass
    
    def close(self,*args):
        pass


if c.live_visualization:
    
    from torch.utils.tensorboard import SummaryWriter
    class LiveVisualizer(Visualizer):
        def __init__(self, loss_labels,stat_labels):
            super().__init__(loss_labels,stat_labels)
            import socket
            from datetime import datetime
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            self.modes = ["train","test"]+dc.viz_years#,"2014","2015","2016","2017","2018","2019"]
            self.fig, self.axes = plt.subplots(n_plots, n_plots, figsize=figsize)
            
            logdir = "tmp/log_dir/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            self.file_writer = tf.summary.create_file_writer(logdir)

            #self.writer_train,self.writer_test,self.writer_2014,self.writer_2015 = None, None, None, None
            #self.writer_2016,self.writer_2017,self.writer_2018,self.writer_2019 = None, None, None, None
            self.writers = {}
            print("viz run_name",c.run_name)
            for _, mode in enumerate(self.modes):
                log_dir = Path(__file__).parent.joinpath('tmp/log_dir/').joinpath(f"{current_time}/{mode}_{c.feature_net_name}_{c.run_name}")
                self.writers[mode] = SummaryWriter(log_dir=log_dir)

            self.state = 0
        def update_losses(self, loss, mode = "train",*args):
            print("update_losses!!!!")
            super().update_losses(loss)

            print("loss", mode, self.state, loss)
            try:
                self.writers[mode].add_scalar(f"loss", loss, self.state)

            except:
                print(f"\n\n\n lossERROR: MODE {mode} NOT FOUND! \n\n\n")

        def update_stats(self, offsets, mode, *args):
            super().update_offsets(offsets)
            for i,offset in enumerate(offsets):
                name = self.stat_labels[i]
                print("update_stats", mode, name, self.state, offset)
                try:
                    self.writers[mode].add_scalar(f"{name}", offset, self.state)
                except Exception as e:
                    print(f"\n\n\n statsERROR: MODE {mode} NOT FOUND! \n\n\n")
                    print(str(e))


        def update_std(self, std, mode, name, *args):
           
            print("update_std", mode, name, self.state, std)
            try:
                self.writers[mode].add_scalar(f"{name}", std, self.state)
            except Exception as e:
                print(f"\n\n\n statsERROR: MODE {mode} NOT FOUND! \n\n\n")
                print(str(e))

        def gen_plot(self,data):
            """Create a pyplot plot and save to buffer."""
            #plt.figure()
            #plt.plot([1, 2])
            #plt.title("test")
            plt.figure("process")

            for i in range(n_plots):
                for j in range(n_plots):
                    try:
                        ax = plt.subplot(n_plots,n_plots, j+n_plots*i+1)
                        ax.clear()
                        if i*n_plots + j<c.x_dim:
                            ax.hist(data[:, i*n_plots + j], bins=20, histtype='step')
                            #self.axes[i,j].clear()
                            #self.axes[i,j].hist(data[:, i*n_plots + j], bins=20, histtype='step')
                            print("plotted ",i,j)
                        else:
                            print("not plotted ",i,j)

                    except ValueError:
                        pass
            buf = io.BytesIO()
            #plt.show()
            plt.savefig(buf, format='png')
            buf.seek(0)
            return buf

        def update_hist(self, data):
            #for i in range(n_plots):
            #    for j in range(n_plots):
            #        try:
            #            self.axes[i,j].clear()
            #            self.axes[i,j].hist(data[:, i*n_plots + j], bins=20, histtype='step')
            #            print("plotted ",i,j)
            #        except ValueError:
            #            pass

            # Prepare the plot
            plot_buf = self.gen_plot(data)

            # Convert PNG buffer to TF image
            image = tf.image.decode_png(plot_buf.getvalue(), channels=4)

            # Add the batch dimension
            image = tf.expand_dims(image, 0)


            # Add image summary
            with self.file_writer.as_default():
                summary_op = tf.summary.image("plot", image, step=self.state)

            # Session
            #with tf.Session() as sess:
                # Run
            #    summary = sess.run(summary_op)
                # Write summary
                
            #self.writers["train"].add_summary(summary_op)
            #tf.summary.image("Training data", img
            

        def make_step(self, step_size = 1):
            self.state += step_size

        def create_graph(self,model, images, mode = "test"):
            self.writers[mode].add_graph(model, images)

        
        def close(self):
            for mode in (self.modes):
                self.writers[mode].close()

    visualizer = LiveVisualizer(loss_names,stat_names)
else:
    visualizer = Visualizer(loss_names,stat_names)


def show_loss(loss, mode = "train"):
    visualizer.update_losses(loss, mode)

def make_step(step_size = 1):
    visualizer.make_step(step_size)

def show_stats(offsets, mode = "train"):
    visualizer.update_stats(offsets, mode)

def show_std(std, mode = "train", name = ""):
    visualizer.update_std(std, mode, name)

def show_graph(model, y_inputs, mode = "test"):
    visualizer.create_graph(model, y_inputs, mode)

def signal_start():
    visualizer.update_running(True)

def signal_stop():
    visualizer.update_running(False)

def show_hist(data):
    visualizer.update_hist(data)

def close():
    visualizer.close()
