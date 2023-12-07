# 在联邦学习中基于本地差分隐私（LDP）的主动成员推断攻击（AMI）

个人学习用

forked from：
https://github.com/trucndt/ami

论文：
https://arxiv.org/abs/2302.12685

## 依赖项
此代码库仅在python 3.8.10和pytorch 1.7.0上进行开发和测试，使用的操作系统为Linux 64位。

### conda
作者准备了一个包含与此项目使用的相同环境规范的文件。要复现此环境（仅适用于Linux 64位操作系统），执行以下命令：

```bash
$ conda create --name <name_env> --file spec-list.txt
```

- `name_env` 是环境的名称

查看所有环境：

```bash
$ conda info --envs
```

使用以下命令激活创建的环境：

```bash
$ conda activate <name_env>
```

## 预处理

1. 遵循 [data_celebA/celeba](data_celebA/celeba), [data_imgnet/](data_imgnet/)中的`README.md`文件来下载数据集
2. 运行 `$ python preprocessing.py`
nohup python preprocessing.py > preprocessing.log 2>&1 &
## 使用方法

- `noldp`：没有差分隐私的AMI攻击
- `ldp`：在差分隐私下的AMI攻击（包括BitRand和OME）

按照这些目录中的`README.md`文件的说明进行操作。

## Disclaimer

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.