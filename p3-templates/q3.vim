let SessionLoad = 1
let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd ~/Notes/AI+ML/10703/hw3/p3-templates
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
argglobal
%argdel
$argadd GCBC.py
edit GCBC.py
let s:save_splitbelow = &splitbelow
let s:save_splitright = &splitright
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
wincmd _ | wincmd |
split
1wincmd k
wincmd w
let &splitbelow = s:save_splitbelow
let &splitright = s:save_splitright
wincmd t
let s:save_winminheight = &winminheight
let s:save_winminwidth = &winminwidth
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 117 + 96) / 193)
exe '2resize ' . ((&lines * 20 + 22) / 45)
exe 'vert 2resize ' . ((&columns * 75 + 96) / 193)
exe '3resize ' . ((&lines * 20 + 22) / 45)
exe 'vert 3resize ' . ((&columns * 75 + 96) / 193)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=4
setlocal fml=1
setlocal fdn=4
setlocal fen
144
normal! zo
let s:l = 157 - ((32 * winheight(0) + 20) / 41)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 157
normal! 021|
lcd ~/Notes/AI+ML/10703/hw3/p3-templates
wincmd w
argglobal
if bufexists("~/Notes/AI+ML/10703/hw3/.gitignore") | buffer ~/Notes/AI+ML/10703/hw3/.gitignore | else | edit ~/Notes/AI+ML/10703/hw3/.gitignore | endif
if &buftype ==# 'terminal'
  silent file ~/Notes/AI+ML/10703/hw3/.gitignore
endif
balt ~/Notes/AI+ML/10703/hw3/p3-templates/playground.py
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=4
setlocal fen
let s:l = 12 - ((11 * winheight(0) + 10) / 20)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 12
normal! 05|
wincmd w
argglobal
if bufexists("term://~/Notes/AI+ML/10703/hw3/p3-templates//295677:/usr/bin/ipython") | buffer term://~/Notes/AI+ML/10703/hw3/p3-templates//295677:/usr/bin/ipython | else | edit term://~/Notes/AI+ML/10703/hw3/p3-templates//295677:/usr/bin/ipython | endif
if &buftype ==# 'terminal'
  silent file term://~/Notes/AI+ML/10703/hw3/p3-templates//295677:/usr/bin/ipython
endif
balt ~/Notes/AI+ML/10703/hw3/p3-templates/playground.py
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=4
setlocal fen
let s:l = 2805 - ((19 * winheight(0) + 10) / 20)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 2805
normal! 0
wincmd w
exe 'vert 1resize ' . ((&columns * 117 + 96) / 193)
exe '2resize ' . ((&lines * 20 + 22) / 45)
exe 'vert 2resize ' . ((&columns * 75 + 96) / 193)
exe '3resize ' . ((&lines * 20 + 22) / 45)
exe 'vert 3resize ' . ((&columns * 75 + 96) / 193)
tabnext 1
badd +157 ~/Notes/AI+ML/10703/hw3/p3-templates/GCBC.py
badd +16 ~/Notes/AI+ML/10703/hw3/p3-templates/playground.py
badd +1 ~/Builds/gym/gym
badd +0 ~/Notes/AI+ML/10703/hw3/.gitignore
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0 && getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20 shortmess=filnxtToOFc
let &winminheight = s:save_winminheight
let &winminwidth = s:save_winminwidth
let s:sx = expand("<sfile>:p:r")."x.vim"
if filereadable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &g:so = s:so_save | let &g:siso = s:siso_save
set hlsearch
nohlsearch
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
