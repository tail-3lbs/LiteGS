echo "Starting 3DGS evaluation across multiple branches..."
echo "Timestamp: $(date)"
echo ""

echo "[1/6] Evaluating branch: dev"
gco dev
pip install litegs/submodules/gaussian_raster
python ./full_eval.py --mipnerf360 /home/jiaqi/workspace/3dgrut/3dgrut/data/mipnerf360
echo "Completed dev branch evaluation"
echo ""

echo "[2/6] Evaluating branch: dev3"
gco dev3
pip install litegs/submodules/gaussian_raster
python ./full_eval.py --mipnerf360 /home/jiaqi/workspace/3dgrut/3dgrut/data/mipnerf360
echo "Completed dev3 branch evaluation"
echo ""

echo "[3/6] Evaluating branch: merge_dash"
gco merge_dash
pip install litegs/submodules/gaussian_raster
python ./full_eval.py --mipnerf360 /home/jiaqi/workspace/3dgrut/3dgrut/data/mipnerf360
echo "Completed merge_dash branch evaluation"
echo ""

echo "[4/6] Evaluating branch: merge_dash_3"
gco merge_dash_3
pip install litegs/submodules/gaussian_raster
python ./full_eval.py --mipnerf360 /home/jiaqi/workspace/3dgrut/3dgrut/data/mipnerf360
echo "Completed merge_dash_3 branch evaluation"
echo ""

echo "[5/6] Evaluating branch: dropout_tiles"
gco dropout_tiles
pip install litegs/submodules/gaussian_raster
python ./full_eval.py --mipnerf360 /home/jiaqi/workspace/3dgrut/3dgrut/data/mipnerf360
echo "Completed dropout_tiles branch evaluation"
echo ""

echo "[6/6] Evaluating branch: dropout_tiles_3"
gco dropout_tiles_3
pip install litegs/submodules/gaussian_raster
python ./full_eval.py --mipnerf360 /home/jiaqi/workspace/3dgrut/3dgrut/data/mipnerf360
echo "Completed dropout_tiles_3 branch evaluation"
echo ""
echo "All evaluations completed successfully at $(date)"
