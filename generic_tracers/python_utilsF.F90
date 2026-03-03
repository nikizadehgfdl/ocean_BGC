#ifdef use_PYTHON
!Inspired by https://github.com/wangsl/python-embedding
!
module python_utilsF
implicit none
public pyF_1_3d
public pyF_1_1d,pyF_2_1d
public pyF_1_2d

contains

subroutine pyF_1_1d(pyScript, pyFunction, varname, x)
  character(len=*),       intent(in) :: pyScript, pyFunction,varname
  real, dimension(:), intent(inout) :: x 
  integer :: n1, len_pyScript, len_pyFunction , len_varname   

  len_pyScript = Len_Trim(pyScript)
  if(pyScript(len_pyScript-2:len_pyScript) .eq. '.py') len_pyScript = len_pyScript-3   
  len_pyFunction = Len_Trim(pyFunction)   
  len_varname  = Len_Trim(varname)   
  n1 = size(x,1)
print*,'pyF_1_1d ',n1
  call pyc_1array1d(pyScript, len_pyScript, pyFunction, len_pyFunction,&
                       varname, len_varname,  x, n1)
      
end subroutine pyF_1_1d

subroutine pyF_2_1d(pyScript, pyFunction, varname, x1, x2)
  character(len=*),       intent(in) :: pyScript, pyFunction,varname
  real, dimension(:), intent(inout) :: x1,x2 
  integer :: n1,n2, len_pyScript, len_pyFunction , len_varname   

  len_pyScript = Len_Trim(pyScript)
  if(pyScript(len_pyScript-2:len_pyScript) .eq. '.py') len_pyScript = len_pyScript-3   
  len_pyFunction = Len_Trim(pyFunction)   
  len_varname  = Len_Trim(varname)   
  n1 = size(x1,1)
  n2 = size(x2,1)
print*,'pyF_2_1d ',n1,n2
  call pyc_2array1d(pyScript, len_pyScript, pyFunction, len_pyFunction,&
                       varname, len_varname,  x1, n1, x2, n2)
      
end subroutine pyF_2_1d

subroutine pyF_1_3d(pyScript, pyFunction, varname, x)
  character(len=*),       intent(in) :: pyScript, pyFunction,varname
  real, dimension(:,:,:), intent(inout) :: x 
  integer :: n1,n2,n3, len_pyScript, len_pyFunction , len_varname   

  len_pyScript = Len_Trim(pyScript)
  if(pyScript(len_pyScript-2:len_pyScript) .eq. '.py') len_pyScript = len_pyScript-3   
  len_pyFunction = Len_Trim(pyFunction)   
  len_varname  = Len_Trim(varname)   

  n1 = size(x,1)
  n2 = size(x,2)       
  n3 = size(x,3)       
  call pyc_1array3d(pyScript, len_pyScript, pyFunction, len_pyFunction,&
                       varname, len_varname,  x, n1,n2,n3)
      
end subroutine pyF_1_3d

subroutine pyF_1_2d(pyScript, pyFunction, varname, x)
  character(len=*),       intent(in) :: pyScript, pyFunction,varname
  real, dimension(:,:), intent(inout) :: x 
  integer :: n1,n2, len_pyScript, len_pyFunction , len_varname   

  len_pyScript = Len_Trim(pyScript)
  if(pyScript(len_pyScript-2:len_pyScript) .eq. '.py') len_pyScript = len_pyScript-3   
  len_pyFunction = Len_Trim(pyFunction)   
  len_varname  = Len_Trim(varname)   

  n1 = size(x,1)
  n2 = size(x,2)              
  call pyc_1array2d(pyScript, len_pyScript, pyFunction, len_pyFunction,&
                       varname, len_varname,  x, n1,n2)
      
end subroutine pyF_1_2d

end module python_utilsF
#endif
