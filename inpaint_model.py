from network import *
from metrics import *
from loss import *


class WNet(object):
    def __init__(self, config):
        self.config = config
        self.base_channel = config.BASE_CHANNEL
        self.sample_num = config.SAMPLE_NUM
        self.exp_base = config.EXPBASE
        self.gamma = config.GAMMA
        self.model_name = 'inpaint'
        self.psnr = PSNR(255.0)
        self.gen_optimizer = tf.train.AdamOptimizer(
            learning_rate=float(config.LR),
            beta1=float(config.BETA1),
            beta2=float(config.BETA2)
        )
        self.dis_optimizer = tf.train.AdamOptimizer(
            learning_rate=float(config.LR) * float(config.D2G_LR),
            beta1=float(config.BETA1),
            beta2=float(config.BETA2)
        )

    def build_whole_model(self, images, masks):
        # normalization [0, 255] to [0, 1]
        images = images / 255
        masks = masks / 255

        # masked
        images_masked = (images * (1 - masks)) + masks

        # inpainting
        outputs, gen_loss, dis_loss = self.inpaint_model(images, images_masked, masks)
        outputs_merged = (outputs * masks) + (images * (1 - masks))

        # recover [0, 1] to [0, 255]
        images = images * 255
        images_masked = images_masked * 255
        outputs_merged = outputs_merged * 255
        outputs = outputs * 255

        # summary
        whole_image = tf.concat([images, images_masked, outputs, outputs_merged], axis=2)
        psnr = self.psnr(images, outputs_merged)
        tf.summary.image('train_image', whole_image, max_outputs=10)
        tf.summary.scalar('psnr', psnr)

        return gen_loss, dis_loss, psnr

    def build_validation_model(self, images, masks):
        # normalization [0, 255] to [0, 1]
        images = images / 255
        masks = masks / 255

        # masked
        images_masked = (images * (1 - masks)) + masks
        inputs = tf.concat([images_masked, masks], axis=3)

        outputs = self.wnet_generator(inputs, self.base_channel, self.sample_num, masks, reuse=True)
        outputs_merged = (outputs * masks) + (images * (1 - masks))

        pred_masks, annotations, weight = self.discriminator(outputs, reuse=True)
        weight = tf.pow(tf.constant(float(self.exp_base)), weight)

        # mask hole ratio
        hr = tf.reduce_sum(masks, axis=[1, 2, 3]) / (256 * 256)

        # calculate validation loss
        gen_loss = 0
        dis_loss = 0

        with tf.variable_scope('validation_loss'):
            # discriminator loss
            dis_seg_loss = focal_loss(annotations, masks, hr, self.gamma)
            dis_loss += dis_seg_loss

            # generator l1 loss
            mask_mean = tf.metrics.mean(masks)
            gen_l1_loss = l1_loss(outputs, images) / mask_mean[1]
            gen_weighted_loss = l1_loss(weight * outputs, weight * images)
            gen_loss += gen_weighted_loss

        # recover [0, 1] to [0, 255]
        images = images * 255
        images_masked = images_masked * 255
        outputs_merged = outputs_merged * 255
        outputs = outputs * 255

        # summary
        whole_image = tf.concat([images, images_masked, outputs, outputs_merged], axis=2)
        tf.summary.image('validation_image', whole_image, max_outputs=10)
        psnr = self.psnr(images, outputs_merged)

        return gen_l1_loss, gen_weighted_loss, dis_loss, psnr

    def build_optim(self, gen_loss, dis_loss):
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.model_name + '_generator')
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.model_name + '_discriminator')
        g_gradient = self.gen_optimizer.compute_gradients(gen_loss, var_list=g_vars)
        d_gradient = self.dis_optimizer.compute_gradients(dis_loss, var_list=d_vars)

        return self.gen_optimizer.apply_gradients(g_gradient), self.dis_optimizer.apply_gradients(d_gradient)

    def inpaint_model(self, images, images_masked, masks):
        # input model
        inputs = tf.concat([images_masked, masks], axis=3)

        # process outputs
        output = self.wnet_generator(inputs, self.base_channel, self.sample_num, masks)
        merged = (output * masks) + (images * (1 - masks))
        gen_loss = 0
        dis_loss = 0

        # create discriminator
        prediction, annotations, weight = self.discriminator(output)
        weight = tf.pow(tf.constant(float(self.exp_base)), weight)

        # mask hole ratio
        hr = tf.reduce_sum(masks, axis=[1, 2, 3]) / (256 * 256)

        with tf.variable_scope('inpaint_loss'):
            # discriminator loss
            dis_seg_loss = focal_loss(annotations, masks, hr, self.gamma)
            dis_loss += dis_seg_loss

            # generator l1 loss
            mask_mean = tf.metrics.mean(masks)
            gen_l1_loss = l1_loss(output, images) / mask_mean[1]
            gen_weighted_loss = l1_loss(weight * output, weight * images)
            gen_content_loss = perceptual_loss(images, output) + perceptual_loss(images, merged)
            gen_loss += 0.05 * gen_content_loss
            gen_loss += gen_weighted_loss

        # summary all of loss
        tf.summary.scalar('loss/dis_loss', dis_loss)
        tf.summary.scalar('loss/gen_weighted_loss', gen_weighted_loss)
        tf.summary.scalar('loss/gen_l1_loss', gen_l1_loss)

        return output, gen_loss, dis_loss


    def wnet_generator(self, x, channel, sample, masks, reuse=False):
        with tf.variable_scope('inpaint_generator', reuse=reuse):
            conv_x = []

            # Down-Sampling
            for i in range(1, sample + 1):
                with tf.variable_scope('encoder_downsample_' + str(i)):
                    x = conv(x, channel, kernel=4, stride=2, pad=1, use_bias=True, scope='conv')
                    x = instance_norm(x, scope='ins_norm')
                    x = relu(x)
                    conv_x.append(x)
                    if i < 4:
                        channel = channel * 2

            coarse_x = []
            # Up-Sampling
            for i in range(sample, 4, -1):
                with tf.variable_scope('encoder_upsample_' + str(i)):
                    coarse_x.append(x)
                    if i != sample:
                        x = x + conv_x[i - 1]
                    x = deconv(x, channel, kernel=4, stride=2, use_bias=True, scope='deconv')
                    x = instance_norm(x, scope='ins_norm')
                    x = relu(x)

            x_in = x
            conv_y = []
            for i in range(3, 0, -1):
                channels = channel // pow(2, 4 - i)
                x = deconv(x, channels, kernel=4, stride=2, use_bias=True, scope='up_' + str(i))
                y, _ = attention(conv_x[i - 1], x, channels, masks, 'attention_' + str(i))           
                conv_y.append(y)

            for i in range(1, 4, 1):
                channels = channel // pow(2, 3 - i)
                if i == 1:
                    y = conv_y[3 - i]
                else:
                    y = y + conv_y[3 - i]
                y = conv(y, channels, kernel=4, stride=2, pad=1, use_bias=True, scope='down_' + str(i))

            x = se_newblock(tf.abs(y - x_in), y, reuse=reuse)
            conv_x = [x]
            coarse_x.reverse()

            # Down-Sampling
            for i in range(5, sample + 1):
                with tf.variable_scope('encoder_downsample_' + str(i), reuse=True):
                    x = conv(x, channel, kernel=4, stride=2, pad=1, use_bias=True, scope='conv')
                    x = instance_norm(x, scope='ins_norm')
                    x = relu(x)
                x = se_newblock(tf.abs(x - coarse_x[i - 5]), x, reuse=True)
                conv_x.append(x)

            # Up-Sampling
            for i in range(sample, 0, -1):
                if i < 5:
                    subreuse = reuse
                else:
                    subreuse = True
                with tf.variable_scope('encoder_upsample_' + str(i), reuse=subreuse):
                    if i < 4:
                        x = tf.concat([x, conv_y[3 - i]], -1)
                        x = conv(x, channel, kernel=3, stride=1, pad=1, scope='conv')
                        x = relu(x)
                    if i < 5:
                        channel = channel // 2
                    if i != sample and i > 3:
                        x = x + conv_x[i - 4]
                    if i != 1:
                        x = deconv(x, channel, kernel=4, stride=2, use_bias=True, scope='deconv')
                        x = instance_norm(x, scope='ins_norm')
                        x = relu(x)
                    else:
                        x = deconv(x, 3, kernel=4, stride=2, use_bias=True, scope='deconv')
                        x = (tf.nn.tanh(x) + 1) / 2

            return x


    def discriminator(self, x, layer=2, reuse=False):
        with tf.variable_scope('inpaint_discriminator', reuse=reuse):
            conv1 = tf.layers.conv2d(x, 32, kernel_size=4, strides=1, padding='SAME', name='conv1')
            conv1 = lrelu(conv1, alpha=0.2)
            conv2 = tf.layers.conv2d(conv1, 64, kernel_size=4, strides=1, padding='SAME', name='conv2')
            conv2 = lrelu(conv2, alpha=0.2)
            conv3 = tf.layers.conv2d(conv2, 128, kernel_size=4, strides=2, padding='SAME', name='conv3')
            conv3 = lrelu(conv3, alpha=0.2)
            conv4 = tf.layers.conv2d(conv3, 256, kernel_size=4, strides=2, padding='SAME', name='conv4')
            conv4 = lrelu(conv4, alpha=0.2)
            conv5 = tf.layers.conv2d(conv4, 256, kernel_size=4, strides=1, padding='SAME', name='conv5')
            conv5 = lrelu(conv5, alpha=0.2)
            x = deconv(conv5, 128, kernel=4, stride=2, use_bias=True, scope='deconv_1')
            x = deconv(x, layer, kernel=4, stride=2, use_bias=True, scope='deconv_2')
            output = tf.cast(tf.argmax(x, dimension=3, name="prediction"), dtype=tf.float32)
            map = tf.nn.softmax(x, axis=-1)
            output = tf.concat([output, map[:, :, :, 1]], axis=2)
            output = tf.expand_dims(output, dim=-1)
        return output, x, tf.expand_dims(map[:, :, :, 1], dim=-1)
