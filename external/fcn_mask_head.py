import numpy as np
import torch
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.mask_heads.fcn_mask_head import (FCNMaskHead,
                                                             _do_paste_mask)

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit


@HEADS.register_module()
class InstanceMaskHead(FCNMaskHead):

    def __init__(self, **kwargs):
        super(InstanceMaskHead, self).__init__(**kwargs)

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        The only difference from InstanceMaskHead and FCNMaskHead is the output
        format of instance masks. The original FCNMaskHead return numpy masks.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape (Tuple): original image height and width, shape (2,)
            scale_factor(float | Tensor): If ``rescale is True``, box
                coordinates are divided by this scale factor to fit
                ``ori_shape``.
            rescale (bool): If True, the resulting masks will be rescaled to
                ``ori_shape``.

        Returns:
            list[list]: encoded masks. The c-th item in the outer list
                corresponds to the c-th class. Given the c-th outer list, the
                i-th item in that inner list is the mask for the i-th box with
                class label c.

        Example:
            >>> import mmcv
            >>> from mmdet.models.roi_heads.mask_heads.fcn_mask_head import *  # NOQA
            >>> N = 7  # N = number of extracted ROIs
            >>> C, H, W = 11, 32, 32
            >>> # Create example instance of FCN Mask Head.
            >>> self = FCNMaskHead(num_classes=C, num_convs=0)
            >>> inputs = torch.rand(N, self.in_channels, H, W)
            >>> mask_pred = self.forward(inputs)
            >>> # Each input is associated with some bounding box
            >>> det_bboxes = torch.Tensor([[1, 1, 42, 42 ]] * N)
            >>> det_labels = torch.randint(0, C, size=(N,))
            >>> rcnn_test_cfg = mmcv.Config({'mask_thr_binary': 0, })
            >>> ori_shape = (H * 4, W * 4)
            >>> scale_factor = torch.FloatTensor((1, 1))
            >>> rescale = False
            >>> # Encoded masks are a list for each category.
            >>> encoded_masks = self.get_seg_masks(
            >>>     mask_pred, det_bboxes, det_labels, rcnn_test_cfg, ori_shape,
            >>>     scale_factor, rescale
            >>> )
            >>> assert len(encoded_masks) == C
            >>> assert sum(list(map(len, encoded_masks))) == N
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid()
        else:
            mask_pred = det_bboxes.new_tensor(mask_pred)

        device = mask_pred.device
        bboxes = det_bboxes[:, :4]
        labels = det_labels

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            if isinstance(scale_factor, float):
                img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
                img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            else:
                w_scale, h_scale = scale_factor[0], scale_factor[1]
                img_h = np.round(ori_shape[0] * h_scale.item()).astype(
                    np.int32)
                img_w = np.round(ori_shape[1] * w_scale.item()).astype(
                    np.int32)
            scale_factor = 1.0

        if not isinstance(scale_factor, (float, torch.Tensor)):
            scale_factor = bboxes.new_tensor(scale_factor)
        bboxes = bboxes / scale_factor

        if torch.onnx.is_in_onnx_export():
            # TODO: Remove after F.grid_sample is supported.
            from torchvision.models.detection.roi_heads \
                import paste_masks_in_image
            masks = paste_masks_in_image(mask_pred, bboxes, ori_shape[:2])
            thr = rcnn_test_cfg.get('mask_thr_binary', 0)
            if thr > 0:
                masks = masks >= thr
            return masks

        N = len(mask_pred)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if device.type == 'cpu':
            # CPU is most efficient when they are pasted one by one with
            # skip_empty=True, so that it performs minimal number of
            # operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks,
            # but may have memory issue
            num_chunks = int(
                np.ceil(N * img_h * img_w * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
            assert (num_chunks <=
                    N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

        threshold = rcnn_test_cfg.mask_thr_binary
        im_mask = torch.zeros(
            N,
            img_h,
            img_w,
            device=device,
            dtype=torch.bool if threshold >= 0 else torch.uint8)

        if not self.class_agnostic:
            mask_pred = mask_pred[range(N), labels][:, None]

        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_pred[inds],
                bboxes[inds],
                img_h,
                img_w,
                skip_empty=device.type == 'cpu')

            if threshold >= 0:
                masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
            else:
                # for visualization and debugging
                masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

            im_mask[(inds, ) + spatial_inds] = masks_chunk

        return im_mask
